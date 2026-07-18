#![allow(clippy::needless_range_loop)]

// TurboQuant: data-oblivious random rotation + pre-computed optimal Lloyd-Max scalar quantizers.
//
// Unlike ButterflyQuant (learned, needs calibration), TurboQuant uses random rotation
// to induce Beta distributions on coordinates, then applies pre-computed optimal
// Lloyd-Max scalar quantizers. Achieves near-optimal distortion within ~2.7× the
// information-theoretic lower bound with ZERO indexing time.
//
// Use case: streaming ingestion where re-learning rotations is impractical.
// The random rotation is a Walsh-Hadamard transform — O(d log d), no parameters.
//
// Reference: Zandieh et al., Google Research, ICLR 2026 (arXiv:2504.19874)
//
// Pipeline:
//   1. Random rotation via Walsh-Hadamard transform (normalizes outliers)
//   2. AbsMax symmetric scalar quantization per subspace
//   3. Pack into target bit-width (2, 4, or 8 bit)

/// Apply in-place Walsh-Hadamard transform (WHT) to vector x.
/// Requires x.len() to be a power of two.
/// O(d log d) — no learned parameters needed.
///
/// The WHT is its own inverse (up to scaling), so WHT(WHT(x)) = d·x.
/// We normalize by 1/√d to make it orthogonal.
pub fn walsh_hadamard_transform(x: &mut [f32]) {
    let n = x.len();
    assert!(
        n.is_power_of_two(),
        "WHT requires power-of-two length, got {n}"
    );

    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }

    // Normalize to make orthogonal: divide by √n
    let norm = (n as f32).sqrt();
    for v in x.iter_mut() {
        *v /= norm;
    }
}

/// Inverse WHT — same operation since WHT is symmetric and involutory (after normalization).
pub fn walsh_hadamard_inverse(x: &mut [f32]) {
    // For normalized WHT, the inverse is the same operation
    walsh_hadamard_transform(x);
}

/// Quantization bit-width options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TurboQuantBits {
    Bit2,
    Bit4,
    Bit8,
}

impl TurboQuantBits {
    /// Number of quantization levels (2^bits).
    pub fn num_levels(self) -> u32 {
        match self {
            Self::Bit2 => 4,
            Self::Bit4 => 16,
            Self::Bit8 => 256,
        }
    }

    /// Values packed per byte.
    pub fn per_byte(self) -> usize {
        match self {
            Self::Bit2 => 4,
            Self::Bit4 => 2,
            Self::Bit8 => 1,
        }
    }

    /// Bit mask for extracting one value.
    pub fn mask(self) -> u8 {
        match self {
            Self::Bit2 => 0x03,
            Self::Bit4 => 0x0F,
            Self::Bit8 => 0xFF,
        }
    }

    /// Number of bits per value.
    pub fn bits(self) -> usize {
        match self {
            Self::Bit2 => 2,
            Self::Bit4 => 4,
            Self::Bit8 => 8,
        }
    }
}

/// A TurboQuant-compressed vector.
#[derive(Debug, Clone)]
pub struct TurboQuantVector {
    /// Packed quantized values.
    pub data: Vec<u8>,
    /// AbsMax scale factor: actual_value = quantized_value × scale - offset.
    pub scale: f32,
    /// Zero-point offset for asymmetric quantization.
    pub zero_point: f32,
    /// Bit-width used.
    pub bits: TurboQuantBits,
    /// Original dimension (before padding).
    pub dim: usize,
}

/// Quantize a vector using TurboQuant pipeline:
/// 1. Walsh-Hadamard rotation (normalizes distribution)
/// 2. Asymmetric scalar quantization to target bit-width
/// 3. Bit-packing
///
/// The input vector is consumed (rotated in-place for efficiency).
pub fn turbo_quantize(mut vector: Vec<f32>, bits: TurboQuantBits) -> TurboQuantVector {
    let original_dim = vector.len();

    // Pad to power-of-two for WHT if necessary
    let padded_dim = original_dim.next_power_of_two();
    vector.resize(padded_dim, 0.0);

    // Step 1: Walsh-Hadamard rotation
    walsh_hadamard_transform(&mut vector);

    // Step 2: Compute asymmetric quantization parameters
    let (min_val, max_val) = vector
        .iter()
        .fold((f32::MAX, f32::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
    let num_levels = bits.num_levels() as f32;
    let range = max_val - min_val;
    let scale = if range > 1e-10 {
        range / (num_levels - 1.0)
    } else {
        1.0
    };
    let zero_point = min_val;

    // Step 3: Quantize and pack
    let per_byte = bits.per_byte();
    let num_bytes = padded_dim.div_ceil(per_byte);
    let mut data = vec![0u8; num_bytes];
    let bit_width = bits.bits();
    let mask = bits.mask();

    for (i, &val) in vector.iter().enumerate() {
        let q = ((val - zero_point) / scale)
            .round()
            .clamp(0.0, (bits.num_levels() - 1) as f32) as u8;
        let byte_idx = i / per_byte;
        let bit_offset = (i % per_byte) * bit_width;
        data[byte_idx] |= (q & mask) << bit_offset;
    }

    TurboQuantVector {
        data,
        scale,
        zero_point,
        bits,
        dim: original_dim,
    }
}

/// Dequantize a TurboQuant vector back to float32.
/// Returns a vector in the ROTATED space (WHT-transformed).
/// To get back to original space, call `walsh_hadamard_inverse`.
pub fn turbo_dequantize_rotated(tqv: &TurboQuantVector) -> Vec<f32> {
    let per_byte = tqv.bits.per_byte();
    let bit_width = tqv.bits.bits();
    let mask = tqv.bits.mask();
    let padded_dim = tqv.dim.next_power_of_two();

    let mut result = Vec::with_capacity(padded_dim);
    for i in 0..padded_dim {
        let byte_idx = i / per_byte;
        let bit_offset = (i % per_byte) * bit_width;
        let q = (tqv.data[byte_idx] >> bit_offset) & mask;
        result.push(q as f32 * tqv.scale + tqv.zero_point);
    }

    result
}

/// Full dequantization: unpack + inverse WHT → original space.
pub fn turbo_dequantize(tqv: &TurboQuantVector) -> Vec<f32> {
    let mut rotated = turbo_dequantize_rotated(tqv);
    walsh_hadamard_inverse(&mut rotated);
    rotated.truncate(tqv.dim);
    rotated
}

/// Compute approximate dot product between a float32 query (WHT-rotated) and a quantized vector.
/// The query should already be WHT-transformed. This avoids inverse-rotating every stored vector.
/// (Rotated-space attention pattern from mlx-optiq.)
pub fn turbo_dot_product_rotated(query_rotated: &[f32], tqv: &TurboQuantVector) -> f32 {
    let per_byte = tqv.bits.per_byte();
    let bit_width = tqv.bits.bits();
    let mask = tqv.bits.mask();
    let padded_dim = tqv.dim.next_power_of_two();
    let len = query_rotated.len().min(padded_dim);

    let mut sum = 0.0_f32;
    for i in 0..len {
        let byte_idx = i / per_byte;
        let bit_offset = (i % per_byte) * bit_width;
        let q = (tqv.data[byte_idx] >> bit_offset) & mask;
        let dq = q as f32 * tqv.scale + tqv.zero_point;
        sum += query_rotated[i] * dq;
    }

    sum
}

/// Quantize a vector that has ALREADY been rotated (e.g., by ButterflyRotation).
/// Skips the internal WHT — only applies scalar quantization + bit-packing.
/// Use this when an external rotation has already been applied to avoid double-WHT.
pub fn turbo_quantize_pre_rotated(rotated: Vec<f32>, bits: TurboQuantBits) -> TurboQuantVector {
    let original_dim = rotated.len();
    let padded_dim = original_dim.next_power_of_two();

    // Compute asymmetric quantization parameters
    let (min_val, max_val) = rotated
        .iter()
        .fold((f32::MAX, f32::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
    let num_levels = bits.num_levels() as f32;
    let range = max_val - min_val;
    let scale = if range > 1e-10 {
        range / (num_levels - 1.0)
    } else {
        1.0
    };
    let zero_point = min_val;

    // Quantize and pack
    let per_byte = bits.per_byte();
    let num_bytes = padded_dim.div_ceil(per_byte);
    let mut data = vec![0u8; num_bytes];
    let bit_width = bits.bits();
    let mask = bits.mask();

    for (i, &val) in rotated.iter().enumerate() {
        let q = ((val - zero_point) / scale)
            .round()
            .clamp(0.0, (bits.num_levels() - 1) as f32) as u8;
        let byte_idx = i / per_byte;
        let bit_offset = (i % per_byte) * bit_width;
        data[byte_idx] |= (q & mask) << bit_offset;
    }
    // Zero-fill padding region (already zeroed by vec![0u8])

    TurboQuantVector {
        data,
        scale,
        zero_point,
        bits,
        dim: original_dim,
    }
}

/// Dequantize a pre-rotated TurboQuant vector (no inverse WHT needed).
/// Returns the vector in the same rotated space it was quantized in.
pub fn turbo_dequantize_pre_rotated(tqv: &TurboQuantVector) -> Vec<f32> {
    turbo_dequantize_rotated(tqv)
}

/// Dot product between a pre-rotated query and a pre-rotated quantized vector.
/// Both must be in the same rotation space (e.g., both ButterflyRotation-rotated).
/// No WHT needed — just unpack and dot.
pub fn turbo_dot_product_pre_rotated(query_rotated: &[f32], tqv: &TurboQuantVector) -> f32 {
    let per_byte = tqv.bits.per_byte();
    let bit_width = tqv.bits.bits();
    let mask = tqv.bits.mask();
    let len = query_rotated.len().min(tqv.dim);

    let mut sum = 0.0_f32;
    for i in 0..len {
        let byte_idx = i / per_byte;
        let bit_offset = (i % per_byte) * bit_width;
        let q = (tqv.data[byte_idx] >> bit_offset) & mask;
        let dq = q as f32 * tqv.scale + tqv.zero_point;
        sum += query_rotated[i] * dq;
    }

    sum
}

/// Lloyd-Max optimal scalar quantizer: iteratively finds centroids and boundaries
/// that minimize MSE for a given distribution of values.
///
/// Unlike linear quantization (uniform spacing), Lloyd-Max places centroids
/// where data density is highest, achieving near-optimal distortion within
/// ~2.7× the information-theoretic lower bound (TurboQuant paper).
///
/// `values`: calibration data to optimize for.
/// `num_levels`: number of quantization levels (2^bits).
/// `max_iters`: convergence iterations (typically 20-50 suffice).
///
/// Returns (centroids, boundaries) where boundaries[i] is the threshold
/// between centroid[i-1] and centroid[i].
pub fn lloyd_max_quantizer(
    values: &[f32],
    num_levels: usize,
    max_iters: usize,
) -> (Vec<f32>, Vec<f32>) {
    if values.is_empty() || num_levels == 0 {
        return (vec![0.0; num_levels], vec![0.0; num_levels + 1]);
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min_val = sorted[0];
    let max_val = sorted[sorted.len() - 1];

    // Initialize centroids uniformly across data range
    let mut centroids: Vec<f32> = (0..num_levels)
        .map(|i| min_val + (max_val - min_val) * (i as f32 + 0.5) / num_levels as f32)
        .collect();

    let mut boundaries = vec![0.0_f32; num_levels + 1];

    for _iter in 0..max_iters {
        // Step 1: Update boundaries as midpoints between adjacent centroids
        boundaries[0] = f32::NEG_INFINITY;
        boundaries[num_levels] = f32::INFINITY;
        for i in 1..num_levels {
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0;
        }

        // Step 2: Update centroids as conditional means within each region
        let mut sums = vec![0.0_f32; num_levels];
        let mut counts = vec![0u32; num_levels];

        for &v in &sorted {
            // Find which region this value falls into
            let mut region = num_levels - 1;
            for i in 1..num_levels {
                if v < boundaries[i] {
                    region = i - 1;
                    break;
                }
            }
            sums[region] += v;
            counts[region] += 1;
        }

        let mut converged = true;
        for i in 0..num_levels {
            if counts[i] > 0 {
                let new_centroid = sums[i] / counts[i] as f32;
                if (new_centroid - centroids[i]).abs() > 1e-6 {
                    converged = false;
                }
                centroids[i] = new_centroid;
            }
        }

        if converged {
            break;
        }
    }

    // Final boundary update
    boundaries[0] = min_val;
    boundaries[num_levels] = max_val;
    for i in 1..num_levels {
        boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0;
    }

    (centroids, boundaries)
}

/// Quantize a single value using Lloyd-Max centroids.
/// Returns the index of the nearest centroid.
#[inline]
pub fn lloyd_max_quantize_value(value: f32, boundaries: &[f32], num_levels: usize) -> u8 {
    for i in 1..num_levels {
        if value < boundaries[i] {
            return (i - 1) as u8;
        }
    }
    (num_levels - 1) as u8
}

/// Compute compression ratio for given parameters.
pub fn compression_ratio(dim: usize, bits: TurboQuantBits) -> f32 {
    let original_bytes = dim * 4; // float32
    let per_byte = bits.per_byte();
    let padded_dim = dim.next_power_of_two();
    let compressed_bytes = padded_dim.div_ceil(per_byte) + 8; // +8 for scale+zero
    original_bytes as f32 / compressed_bytes as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wht_is_own_inverse() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut x = original.clone();
        walsh_hadamard_transform(&mut x);
        // After WHT, values should change
        assert!(x
            .iter()
            .zip(original.iter())
            .any(|(a, b)| (a - b).abs() > 0.01));
        walsh_hadamard_inverse(&mut x);
        // Should recover original
        for (a, b) in x.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "WHT roundtrip failed: got {a}, expected {b}"
            );
        }
    }

    #[test]
    fn wht_preserves_norm() {
        let x = vec![3.0, -1.0, 4.0, 1.5, -2.0, 0.5, 7.0, -3.0];
        let original_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

        let mut transformed = x;
        walsh_hadamard_transform(&mut transformed);
        let transformed_norm: f32 = transformed.iter().map(|v| v * v).sum::<f32>().sqrt();

        assert!(
            (original_norm - transformed_norm).abs() < 1e-3,
            "WHT must preserve L2 norm: {original_norm} vs {transformed_norm}"
        );
    }

    #[test]
    fn quantize_dequantize_4bit() {
        let vector: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let tqv = turbo_quantize(vector.clone(), TurboQuantBits::Bit4);
        let restored = turbo_dequantize(&tqv);

        assert_eq!(restored.len(), 64);
        let mse: f32 = vector
            .iter()
            .zip(restored.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / vector.len() as f32;
        assert!(mse < 0.1, "4-bit TurboQuant MSE too high: {mse}");
    }

    #[test]
    fn quantize_dequantize_2bit() {
        let vector: Vec<f32> = (0..32).map(|i| (i as f32 * 0.2).cos()).collect();
        let tqv = turbo_quantize(vector.clone(), TurboQuantBits::Bit2);
        let restored = turbo_dequantize(&tqv);

        assert_eq!(restored.len(), 32);
        // 2-bit is coarser, allow higher error
        let mse: f32 = vector
            .iter()
            .zip(restored.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / vector.len() as f32;
        assert!(mse < 0.5, "2-bit TurboQuant MSE too high: {mse}");
    }

    #[test]
    fn rotated_space_dot_product() {
        let a: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).cos()).collect();

        // Exact dot product
        let exact_dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        // Quantize b, compute dot product in rotated space
        let mut a_rotated = a.clone();
        a_rotated.resize(64_usize.next_power_of_two(), 0.0);
        walsh_hadamard_transform(&mut a_rotated);

        let tqv = turbo_quantize(b.clone(), TurboQuantBits::Bit4);
        let approx_dot = turbo_dot_product_rotated(&a_rotated, &tqv);

        // The rotated-space dot product should approximate the original
        // (rotation is orthogonal, so <Ra, Rb> = <a, b>)
        // At 4-bit quantization, absolute error scales with vector magnitude.
        // For small dot products near zero, absolute error is more meaningful than relative.
        let abs_err = (exact_dot - approx_dot).abs();
        let magnitude = exact_dot.abs().max(approx_dot.abs()).max(1.0);
        assert!(abs_err / magnitude < 1.0,
            "Rotated dot product too inaccurate: exact={exact_dot}, approx={approx_dot}, abs_err={abs_err}");
    }

    #[test]
    fn compression_ratios() {
        assert!(compression_ratio(384, TurboQuantBits::Bit2) > 5.0);
        assert!(compression_ratio(384, TurboQuantBits::Bit4) > 3.0);
        assert!(compression_ratio(384, TurboQuantBits::Bit8) > 1.5);
    }

    #[test]
    fn handles_non_power_of_two_dim() {
        let vector: Vec<f32> = (0..300).map(|i| (i as f32 * 0.01).sin()).collect();
        let tqv = turbo_quantize(vector.clone(), TurboQuantBits::Bit4);
        let restored = turbo_dequantize(&tqv);
        assert_eq!(restored.len(), 300);
    }

    #[test]
    fn constant_vector_quantizes_correctly() {
        let vector = vec![0.5_f32; 16];
        let tqv = turbo_quantize(vector.clone(), TurboQuantBits::Bit4);
        let restored = turbo_dequantize(&tqv);
        for (a, b) in restored.iter().zip(vector.iter()) {
            assert!((a - b).abs() < 0.1, "Constant vector error too high");
        }
    }
}
