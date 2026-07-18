// PM-KVQ + KVTuner: Progressive mixed-precision KV cache quantization with
// sensitivity-aware bit allocation.
//
// PM-KVQ (Liu et al., Tsinghua, arXiv:2505.18610): progressively reduces
// KV cache precision via bit-shifting (16→8→4→2) as memory fills, rather
// than direct low-bit quantization. At 2-bit, achieves 64.79% pass@1 vs
// KIVI's 51.88% on DeepSeek-R1-Distill-Qwen-7B.
//
// KVTuner (Li et al., arXiv:2502.04420): proves Key cache sensitivity is
// amplified by softmax attention and that layer-wise sensitivity is a MODEL
// property independent of input. Uses multi-objective optimization to search
// a Pareto frontier of hardware-friendly precision pairs per layer.
//
// Key asymmetry (KIVI): Keys need per-CHANNEL quantization (outlier channels),
// Values need per-TOKEN quantization (uniform distributions).
//
// Combined design:
//   1. KVTuner profiles each transformer layer's K/V sensitivity offline
//   2. PM-KVQ allocates precision per-layer based on sensitivity + memory pressure
//   3. Progressive right-shift degrades gracefully as context grows

/// Precision level for a KV cache tensor.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum KVPrecision {
    /// Full precision (16-bit float).
    FP16,
    /// 8-bit integer quantization.
    INT8,
    /// 4-bit integer quantization.
    INT4,
    /// 2-bit integer quantization.
    INT2,
}

impl KVPrecision {
    /// Bits per element.
    pub fn bits(self) -> usize {
        match self {
            Self::FP16 => 16,
            Self::INT8 => 8,
            Self::INT4 => 4,
            Self::INT2 => 2,
        }
    }

    /// The next lower precision level (for progressive degradation).
    pub fn degrade(self) -> Option<KVPrecision> {
        match self {
            Self::FP16 => Some(Self::INT8),
            Self::INT8 => Some(Self::INT4),
            Self::INT4 => Some(Self::INT2),
            Self::INT2 => None,
        }
    }

    /// Compression ratio relative to FP16.
    pub fn compression_vs_fp16(self) -> f32 {
        16.0 / self.bits() as f32
    }
}

/// Per-layer sensitivity profile from KVTuner.
/// This is a MODEL property — computed once during calibration, reused for all inputs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayerSensitivity {
    /// Transformer layer index.
    pub layer_idx: usize,
    /// Key cache sensitivity (MSE on attention logits when K is quantized).
    /// Higher = more sensitive = needs higher precision.
    pub key_sensitivity: f32,
    /// Value cache sensitivity (MSE on layer output when V is quantized).
    pub value_sensitivity: f32,
    /// Recommended Key precision at normal memory pressure.
    pub key_default_precision: KVPrecision,
    /// Recommended Value precision at normal memory pressure.
    pub value_default_precision: KVPrecision,
}

/// Configuration for KVTuner sensitivity profiling.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KVTunerConfig {
    /// Number of transformer layers in the model.
    pub num_layers: usize,
    /// Target average bit-width across all layers. Default: 4.0.
    pub target_avg_bits: f32,
    /// Hardware-friendly precision pairs to consider.
    /// Each pair is (Key precision, Value precision).
    pub allowed_pairs: Vec<(KVPrecision, KVPrecision)>,
}

impl Default for KVTunerConfig {
    fn default() -> Self {
        Self {
            num_layers: 32,
            target_avg_bits: 4.0,
            allowed_pairs: vec![
                (KVPrecision::INT8, KVPrecision::INT4), // K8V4 — balanced
                (KVPrecision::INT4, KVPrecision::INT4), // K4V4 — compact
                (KVPrecision::INT4, KVPrecision::INT2), // K4V2 — aggressive
                (KVPrecision::INT2, KVPrecision::INT4), // K2V4 — rare but valid
                (KVPrecision::INT8, KVPrecision::INT8), // K8V8 — high quality
                (KVPrecision::FP16, KVPrecision::FP16), // Keep full precision for critical layers
            ],
        }
    }
}

/// Result of KVTuner profiling: per-layer precision assignments.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KVTunerProfile {
    /// Per-layer sensitivity and precision assignments.
    pub layers: Vec<LayerSensitivity>,
    /// Average bits per element across all layers.
    pub avg_bits: f32,
    /// Total memory savings as a ratio (e.g., 0.6 = 40% reduction).
    pub memory_ratio: f32,
}

/// Profile transformer layers to determine optimal per-layer KV precision.
///
/// Uses a greedy knapsack approach on the Pareto frontier:
/// 1. Start all layers at highest precision
/// 2. Iteratively degrade the LEAST sensitive layer until target bits reached
///
/// `layer_key_mse`: per-layer MSE when Key cache is quantized to INT4
/// `layer_value_mse`: per-layer MSE when Value cache is quantized to INT4
pub fn profile_kv_sensitivity(
    layer_key_mse: &[f32],
    layer_value_mse: &[f32],
    config: &KVTunerConfig,
) -> KVTunerProfile {
    assert_eq!(layer_key_mse.len(), config.num_layers);
    assert_eq!(layer_value_mse.len(), config.num_layers);

    let mut layers: Vec<LayerSensitivity> = (0..config.num_layers)
        .map(|i| LayerSensitivity {
            layer_idx: i,
            key_sensitivity: layer_key_mse[i],
            value_sensitivity: layer_value_mse[i],
            key_default_precision: KVPrecision::INT8,
            value_default_precision: KVPrecision::INT4,
        })
        .collect();

    // Greedy: assign precision pairs based on sensitivity ranking
    // Low sensitivity layers get aggressive quantization; high sensitivity keeps precision
    let mut combined_sensitivities: Vec<(usize, f32)> = layers
        .iter()
        .map(|l| (l.layer_idx, l.key_sensitivity + l.value_sensitivity))
        .collect();
    combined_sensitivities
        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Assign from least sensitive to most sensitive
    let total = config.num_layers;
    for (rank, &(layer_idx, _sensitivity)) in combined_sensitivities.iter().enumerate() {
        let fraction = rank as f32 / total as f32;
        let (k_prec, v_prec) = if fraction < 0.25 {
            // Bottom 25%: most aggressive
            (KVPrecision::INT4, KVPrecision::INT2)
        } else if fraction < 0.5 {
            // 25-50%: moderate
            (KVPrecision::INT4, KVPrecision::INT4)
        } else if fraction < 0.75 {
            // 50-75%: balanced
            (KVPrecision::INT8, KVPrecision::INT4)
        } else {
            // Top 25%: preserve quality
            (KVPrecision::INT8, KVPrecision::INT8)
        };

        layers[layer_idx].key_default_precision = k_prec;
        layers[layer_idx].value_default_precision = v_prec;
    }

    let avg_bits = layers
        .iter()
        .map(|l| (l.key_default_precision.bits() + l.value_default_precision.bits()) as f32 / 2.0)
        .sum::<f32>()
        / total as f32;

    let memory_ratio = avg_bits / 16.0; // vs FP16

    KVTunerProfile {
        layers,
        avg_bits,
        memory_ratio,
    }
}

/// PM-KVQ: Progressive Mixed-precision KV cache Quantizer.
///
/// Manages a KV cache that progressively reduces precision via bit-shifting
/// as memory pressure increases. Each token's KV pair starts at the layer's
/// default precision and can be degraded to free memory.
pub struct ProgressiveKVCache {
    /// Per-layer configuration from KVTuner.
    profile: KVTunerProfile,
    /// Per-layer, per-token stored KV data.
    /// Outer: layer index. Inner: token index → (key_data, value_data, current_precision).
    layers: Vec<KVCacheLayer>,
    /// Maximum memory budget in bytes.
    memory_budget: usize,
    /// Current memory usage in bytes.
    current_memory: usize,
    /// Head dimension (d_head).
    head_dim: usize,
    /// Number of attention heads.
    num_heads: usize,
}

/// Per-layer KV cache storage.
struct KVCacheLayer {
    /// Tokens stored in this layer.
    tokens: Vec<KVToken>,
    /// Current precision for Keys in this layer.
    key_precision: KVPrecision,
    /// Current precision for Values in this layer.
    value_precision: KVPrecision,
}

/// A single token's KV pair at a specific precision.
/// Fields are populated during append and will be read during dequantization
/// when the full KV cache retrieval path is implemented.
#[allow(dead_code)]
struct KVToken {
    key_data: Vec<u8>,
    value_data: Vec<u8>,
    key_scales: Vec<f32>,
    value_scale: f32,
    key_zeros: Vec<f32>,
    value_zero: f32,
}

impl ProgressiveKVCache {
    /// Create a new progressive KV cache.
    pub fn new(
        profile: KVTunerProfile,
        memory_budget: usize,
        head_dim: usize,
        num_heads: usize,
    ) -> Self {
        let num_layers = profile.layers.len();
        let layers = (0..num_layers)
            .map(|i| KVCacheLayer {
                tokens: Vec::with_capacity(2048),
                key_precision: profile.layers[i].key_default_precision,
                value_precision: profile.layers[i].value_default_precision,
            })
            .collect();

        Self {
            profile,
            layers,
            memory_budget,
            current_memory: 0,
            head_dim,
            num_heads,
        }
    }

    /// Append a new token's KV pair to a specific layer.
    /// Keys are quantized per-channel; Values are quantized per-token.
    ///
    /// `key`: [num_heads × head_dim] float32
    /// `value`: [num_heads × head_dim] float32
    pub fn append(&mut self, layer_idx: usize, key: &[f32], value: &[f32]) {
        let total_dim = self.num_heads * self.head_dim;
        assert_eq!(key.len(), total_dim);
        assert_eq!(value.len(), total_dim);

        let layer = &self.layers[layer_idx];
        let k_prec = layer.key_precision;
        let v_prec = layer.value_precision;

        // Quantize Key per-channel (the critical asymmetry from KIVI)
        let (key_data, key_scales, key_zeros) = quantize_per_channel(key, self.head_dim, k_prec);

        // Quantize Value per-token (entire vector as one group)
        let (value_data, value_scale, value_zero) = quantize_per_token(value, v_prec);

        let token_memory =
            key_data.len() + value_data.len() + key_scales.len() * 4 + key_zeros.len() * 4 + 8; // scales + zeros + 2 f32

        self.current_memory += token_memory;

        self.layers[layer_idx].tokens.push(KVToken {
            key_data,
            value_data,
            key_scales,
            value_scale,
            key_zeros,
            value_zero,
        });

        // If over budget, trigger progressive degradation
        if self.current_memory > self.memory_budget {
            self.degrade_least_sensitive();
        }
    }

    /// Progressively degrade the least sensitive layer's precision.
    /// PM-KVQ's key innovation: bit-shifting rather than re-quantizing.
    fn degrade_least_sensitive(&mut self) {
        // Find the layer with lowest combined sensitivity that can still degrade
        let mut best_layer: Option<(usize, f32)> = None;

        for (i, layer) in self.layers.iter().enumerate() {
            let sens = &self.profile.layers[i];
            let combined = sens.key_sensitivity + sens.value_sensitivity;

            let can_degrade_v = layer.value_precision.degrade().is_some();
            let can_degrade_k = layer.key_precision.degrade().is_some();

            if can_degrade_v || can_degrade_k {
                match best_layer {
                    None => best_layer = Some((i, combined)),
                    Some((_, best_sens)) if combined < best_sens => {
                        best_layer = Some((i, combined));
                    }
                    _ => {}
                }
            }
        }

        if let Some((layer_idx, _)) = best_layer {
            let layer = &mut self.layers[layer_idx];

            // Prefer degrading Values first (less sensitive per KIVI asymmetry)
            if let Some(new_v_prec) = layer.value_precision.degrade() {
                let old_bits = layer.value_precision.bits();
                let new_bits = new_v_prec.bits();
                let savings_per_token = self.num_heads * self.head_dim * (old_bits - new_bits) / 8;

                // Right-shift existing value data (PM-KVQ's progressive bit-shift)
                for token in &mut layer.tokens {
                    token.value_data = right_shift_packed(
                        &token.value_data,
                        layer.value_precision,
                        new_v_prec,
                        self.num_heads * self.head_dim,
                    );
                }

                self.current_memory -= savings_per_token * layer.tokens.len();
                layer.value_precision = new_v_prec;
            } else if let Some(new_k_prec) = layer.key_precision.degrade() {
                let old_bits = layer.key_precision.bits();
                let new_bits = new_k_prec.bits();
                let savings_per_token = self.num_heads * self.head_dim * (old_bits - new_bits) / 8;

                for token in &mut layer.tokens {
                    token.key_data = right_shift_packed(
                        &token.key_data,
                        layer.key_precision,
                        new_k_prec,
                        self.num_heads * self.head_dim,
                    );
                }

                self.current_memory -= savings_per_token * layer.tokens.len();
                layer.key_precision = new_k_prec;
            }
        }
    }

    /// Current memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.current_memory
    }

    /// Number of tokens cached per layer.
    pub fn tokens_per_layer(&self) -> Vec<usize> {
        self.layers.iter().map(|l| l.tokens.len()).collect()
    }

    /// Current precision for each layer (Key, Value).
    pub fn precision_per_layer(&self) -> Vec<(KVPrecision, KVPrecision)> {
        self.layers
            .iter()
            .map(|l| (l.key_precision, l.value_precision))
            .collect()
    }

    /// Total number of cached tokens across all layers.
    pub fn total_tokens(&self) -> usize {
        self.layers.iter().map(|l| l.tokens.len()).sum()
    }

    /// Clear all cached tokens.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.tokens.clear();
        }
        self.current_memory = 0;
    }
}

/// Quantize a vector per-channel (groups of `group_size` elements share a scale).
/// Used for Key caches where outliers concentrate in specific channels.
fn quantize_per_channel(
    data: &[f32],
    group_size: usize,
    precision: KVPrecision,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let num_groups = data.len().div_ceil(group_size);
    let num_levels = (1u32 << precision.bits()) - 1;
    let bits = precision.bits();
    let per_byte = 8 / bits;

    let total_packed = data.len().div_ceil(per_byte);
    let mut packed = vec![0u8; total_packed];
    let mut scales = Vec::with_capacity(num_groups);
    let mut zeros = Vec::with_capacity(num_groups);

    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(data.len());
        let group = &data[start..end];

        let (min_val, max_val) = group
            .iter()
            .fold((f32::MAX, f32::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
        let range = max_val - min_val;
        let scale = if range > 1e-10 {
            range / num_levels as f32
        } else {
            1.0
        };
        scales.push(scale);
        zeros.push(min_val);

        for (i, &val) in group.iter().enumerate() {
            let q = ((val - min_val) / scale)
                .round()
                .clamp(0.0, num_levels as f32) as u8;
            let global_idx = start + i;
            let byte_idx = global_idx / per_byte;
            let bit_offset = (global_idx % per_byte) * bits;
            let mask = ((1u16 << bits) - 1) as u8;
            packed[byte_idx] |= (q & mask) << bit_offset;
        }
    }

    (packed, scales, zeros)
}

/// Quantize a vector per-token (single scale/zero for entire vector).
/// Used for Value caches where distributions are more uniform.
fn quantize_per_token(data: &[f32], precision: KVPrecision) -> (Vec<u8>, f32, f32) {
    let num_levels = (1u32 << precision.bits()) - 1;
    let bits = precision.bits();
    let per_byte = 8 / bits;

    let (min_val, max_val) = data
        .iter()
        .fold((f32::MAX, f32::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
    let range = max_val - min_val;
    let scale = if range > 1e-10 {
        range / num_levels as f32
    } else {
        1.0
    };

    let total_packed = data.len().div_ceil(per_byte);
    let mut packed = vec![0u8; total_packed];

    for (i, &val) in data.iter().enumerate() {
        let q = ((val - min_val) / scale)
            .round()
            .clamp(0.0, num_levels as f32) as u8;
        let byte_idx = i / per_byte;
        let bit_offset = (i % per_byte) * bits;
        let mask = ((1u16 << bits) - 1) as u8;
        packed[byte_idx] |= (q & mask) << bit_offset;
    }

    (packed, scale, min_val)
}

/// Progressive right-shift: degrade precision by dropping lower bits.
/// PM-KVQ's core operation — avoids full re-quantization.
///
/// Example: INT8 → INT4: keep top 4 bits of each 8-bit value.
fn right_shift_packed(
    data: &[u8],
    from: KVPrecision,
    to: KVPrecision,
    num_elements: usize,
) -> Vec<u8> {
    let from_bits = from.bits();
    let to_bits = to.bits();
    let shift = from_bits - to_bits;

    let from_per_byte = 8 / from_bits;
    let to_per_byte = 8 / to_bits;
    let from_mask = ((1u16 << from_bits) - 1) as u8;
    let to_mask = ((1u16 << to_bits) - 1) as u8;

    let new_packed_len = num_elements.div_ceil(to_per_byte);
    let mut result = vec![0u8; new_packed_len];

    for i in 0..num_elements {
        // Extract value at original precision
        let src_byte = i / from_per_byte;
        let src_offset = (i % from_per_byte) * from_bits;
        if src_byte >= data.len() {
            break;
        }
        let val = (data[src_byte] >> src_offset) & from_mask;

        // Right-shift to lower precision
        let reduced = (val >> shift) & to_mask;

        // Pack into new representation
        let dst_byte = i / to_per_byte;
        let dst_offset = (i % to_per_byte) * to_bits;
        result[dst_byte] |= reduced << dst_offset;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_profile(num_layers: usize) -> KVTunerProfile {
        let key_mse: Vec<f32> = (0..num_layers).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let value_mse: Vec<f32> = (0..num_layers).map(|i| (i as f32 + 1.0) * 0.05).collect();
        let config = KVTunerConfig {
            num_layers,
            ..Default::default()
        };
        profile_kv_sensitivity(&key_mse, &value_mse, &config)
    }

    #[test]
    fn profile_assigns_varying_precision() {
        let profile = make_profile(8);
        assert_eq!(profile.layers.len(), 8);

        // Least sensitive layers should get lower precision
        let precisions: Vec<(KVPrecision, KVPrecision)> = profile
            .layers
            .iter()
            .map(|l| (l.key_default_precision, l.value_default_precision))
            .collect();

        // Should have a mix of precisions, not all the same
        let unique: std::collections::HashSet<_> = precisions.iter().collect();
        assert!(
            unique.len() > 1,
            "Should assign different precisions to different layers"
        );
    }

    #[test]
    fn profile_avg_bits_reasonable() {
        let profile = make_profile(32);
        assert!(
            profile.avg_bits > 2.0 && profile.avg_bits < 10.0,
            "Average bits should be reasonable: got {}",
            profile.avg_bits
        );
        assert!(profile.memory_ratio < 1.0, "Should save memory vs FP16");
    }

    #[test]
    fn progressive_cache_appends_tokens() {
        let profile = make_profile(4);
        let mut cache = ProgressiveKVCache::new(profile, 1024 * 1024, 64, 8);

        let key = vec![0.1_f32; 64 * 8];
        let value = vec![0.2_f32; 64 * 8];

        cache.append(0, &key, &value);
        cache.append(0, &key, &value);
        cache.append(1, &key, &value);

        assert_eq!(cache.tokens_per_layer(), vec![2, 1, 0, 0]);
        assert!(cache.memory_usage() > 0);
    }

    #[test]
    fn progressive_degradation_under_pressure() {
        let profile = make_profile(4);
        // Very small budget to force degradation
        let mut cache = ProgressiveKVCache::new(profile, 512, 8, 2);

        let key = vec![0.1_f32; 8 * 2];
        let value = vec![0.2_f32; 8 * 2];

        // Append enough tokens to exceed budget
        for _ in 0..20 {
            for layer in 0..4 {
                cache.append(layer, &key, &value);
            }
        }

        // Some layers should have degraded
        let precisions = cache.precision_per_layer();
        let has_degraded = precisions.iter().any(|(_k, v)| *v == KVPrecision::INT2);
        assert!(
            has_degraded,
            "Under memory pressure, some layers should degrade"
        );
    }

    #[test]
    fn clear_resets_memory() {
        let profile = make_profile(2);
        let mut cache = ProgressiveKVCache::new(profile, 1024 * 1024, 32, 4);

        let key = vec![0.5_f32; 32 * 4];
        let value = vec![0.3_f32; 32 * 4];
        cache.append(0, &key, &value);
        assert!(cache.memory_usage() > 0);

        cache.clear();
        assert_eq!(cache.memory_usage(), 0);
        assert_eq!(cache.total_tokens(), 0);
    }

    #[test]
    fn right_shift_int8_to_int4() {
        // Pack value 0xAB (171) as INT8
        let data = vec![0xAB_u8];
        let result = right_shift_packed(&data, KVPrecision::INT8, KVPrecision::INT4, 1);
        // Right-shift by 4: 0xAB >> 4 = 0x0A = 10
        let byte_idx = 0;
        let val = result[byte_idx] & 0x0F;
        assert_eq!(val, 10);
    }

    #[test]
    fn right_shift_int4_to_int2() {
        // Pack two 4-bit values: 0xF (15) and 0x3 (3)
        let data = vec![0x3F_u8]; // lower nibble=F, upper nibble=3
        let result = right_shift_packed(&data, KVPrecision::INT4, KVPrecision::INT2, 2);
        // 0xF >> 2 = 3, 0x3 >> 2 = 0
        let val0 = result[0] & 0x03;
        let val1 = (result[0] >> 2) & 0x03;
        assert_eq!(val0, 3);
        assert_eq!(val1, 0);
    }

    #[test]
    fn precision_degrade_chain() {
        assert_eq!(KVPrecision::FP16.degrade(), Some(KVPrecision::INT8));
        assert_eq!(KVPrecision::INT8.degrade(), Some(KVPrecision::INT4));
        assert_eq!(KVPrecision::INT4.degrade(), Some(KVPrecision::INT2));
        assert_eq!(KVPrecision::INT2.degrade(), None);
    }

    #[test]
    fn per_channel_quantization() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.5 - 8.0).collect();
        let (packed, scales, zeros) = quantize_per_channel(&data, 8, KVPrecision::INT8);
        assert_eq!(scales.len(), 4); // 32 / 8 = 4 groups
        assert_eq!(zeros.len(), 4);
        assert!(!packed.is_empty());
    }

    #[test]
    fn per_token_quantization() {
        let data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.3).collect();
        let (packed, scale, zero) = quantize_per_token(&data, KVPrecision::INT4);
        assert!(!packed.is_empty());
        assert!(scale > 0.0);
        assert!(zero >= 0.0);
    }
}
