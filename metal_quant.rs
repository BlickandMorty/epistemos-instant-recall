#![allow(clippy::manual_clamp)]

// Metal compute shaders for quantized vector operations on Apple Silicon.
//
// Template-specialized dequantization kernels for 2-bit and 4-bit,
// zero-copy UMA buffer sharing via StorageModeShared, and fused
// dequantize-dot-product compute pipelines.
//
// Architecture:
//   Rust (epistemos-core) creates MTLBuffers with StorageModeShared
//   → zero-copy: both CPU and GPU read/write the same physical memory
//   → no PCIe transfer overhead (UMA advantage over discrete GPUs)
//   → Metal compute shaders operate on quantized data in-place
//
// Kernel specialization:
//   Rather than branching on bit-width at runtime, we compile separate
//   pipelines for 2-bit and 4-bit via Metal function constants.
//   This matches MLX's QuantizedBlockLoader approach.
//
// Uses objc2-metal (NOT deprecated metal-rs).
// Note: objc2-metal requires the objc2 ecosystem and uses safe Rust wrappers.
// When objc2-metal isn't available at compile time (e.g., CI), this module
// compiles as stubs using the `metal_gpu` feature flag.

/// Metal Shading Language source for quantized vector operations.
///
/// Two kernels:
///   1. `dequantize_dot_product_2bit`: fused dequant + dot product for 2-bit vectors
///   2. `dequantize_dot_product_4bit`: fused dequant + dot product for 4-bit vectors
///
/// Each threadgroup processes one vector pair (query × stored_vector).
/// The reduction uses threadgroup shared memory for the partial sums.
pub const METAL_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// 2-bit dequantization + dot product kernel.
// Each thread unpacks 16 elements (4 bytes → 16 × 2-bit values).
//
// Args:
//   query_buffer:  float32 query vector (already in rotated space)
//   quant_buffer:  packed 2-bit quantized vectors (contiguous, one per row)
//   scales_buffer: per-vector scale factors (float32)
//   zeros_buffer:  per-vector zero points (float32)
//   output_buffer: per-vector dot product scores (float32)
//   dim:           padded vector dimension
//   num_vectors:   number of stored vectors
kernel void dequantize_dot_product_2bit(
    device const float*   query_buffer   [[buffer(0)]],
    device const uint8_t* quant_buffer   [[buffer(1)]],
    device const float*   scales_buffer  [[buffer(2)]],
    device const float*   zeros_buffer   [[buffer(3)]],
    device float*         output_buffer  [[buffer(4)]],
    constant uint&        dim            [[buffer(5)]],
    constant uint&        num_vectors    [[buffer(6)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    if (gid >= num_vectors) return;

    uint bytes_per_vec = (dim + 3) / 4;  // 4 values per byte at 2-bit
    device const uint8_t* vec_data = quant_buffer + gid * bytes_per_vec;
    float scale = scales_buffer[gid];
    float zero  = zeros_buffer[gid];

    // Each thread computes a partial sum over its assigned elements
    float partial_sum = 0.0;

    for (uint i = tid; i < dim; i += tpg) {
        uint byte_idx = i / 4;
        uint bit_offset = (i % 4) * 2;
        uint8_t q = (vec_data[byte_idx] >> bit_offset) & 0x03;
        float dequantized = float(q) * scale + zero;
        partial_sum += query_buffer[i] * dequantized;
    }

    // Threadgroup reduction (warp-level reduce for Apple GPU)
    // Apple GPUs have 32-wide SIMD groups
    threadgroup float shared_sums[256];
    shared_sums[tid] = partial_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] += shared_sums[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output_buffer[gid] = shared_sums[0];
    }
}

// 4-bit dequantization + dot product kernel.
// Each thread unpacks 8 elements (4 bytes → 8 × 4-bit values).
kernel void dequantize_dot_product_4bit(
    device const float*   query_buffer   [[buffer(0)]],
    device const uint8_t* quant_buffer   [[buffer(1)]],
    device const float*   scales_buffer  [[buffer(2)]],
    device const float*   zeros_buffer   [[buffer(3)]],
    device float*         output_buffer  [[buffer(4)]],
    constant uint&        dim            [[buffer(5)]],
    constant uint&        num_vectors    [[buffer(6)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    if (gid >= num_vectors) return;

    uint bytes_per_vec = (dim + 1) / 2;  // 2 values per byte at 4-bit
    device const uint8_t* vec_data = quant_buffer + gid * bytes_per_vec;
    float scale = scales_buffer[gid];
    float zero  = zeros_buffer[gid];

    float partial_sum = 0.0;

    for (uint i = tid; i < dim; i += tpg) {
        uint byte_idx = i / 2;
        uint bit_offset = (i % 2) * 4;
        uint8_t q = (vec_data[byte_idx] >> bit_offset) & 0x0F;
        float dequantized = float(q) * scale + zero;
        partial_sum += query_buffer[i] * dequantized;
    }

    threadgroup float shared_sums[256];
    shared_sums[tid] = partial_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] += shared_sums[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output_buffer[gid] = shared_sums[0];
    }
}

// Kitty two-tensor fused dequantize + dot product.
// Uses a precomputed prefix-sum buffer for O(1) boost index lookup
// instead of per-thread O(d) mask scanning.
//
// boost_prefix_sum[i] = number of boosted channels in [0, i).
// Precomputed on CPU and passed as buffer — eliminates all SIMD divergence
// from the boost index calculation.
kernel void dequantize_dot_product_kitty(
    device const float*   query_buffer      [[buffer(0)]],
    device const uint8_t* base_buffer       [[buffer(1)]],
    device const uint8_t* boost_buffer      [[buffer(2)]],
    device const float*   base_scales       [[buffer(3)]],
    device const float*   base_zeros        [[buffer(4)]],
    device const float*   boost_scales      [[buffer(5)]],
    device const float*   boost_zeros       [[buffer(6)]],
    device const uint16_t* boost_prefix_sum [[buffer(7)]],  // prefix sum, not bitmask
    device float*         output_buffer     [[buffer(8)]],
    constant uint&        dim               [[buffer(9)]],
    constant uint&        num_vectors       [[buffer(10)]],
    constant uint&        num_boosted       [[buffer(11)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    if (gid >= num_vectors) return;

    uint base_bytes_per_vec = (dim + 3) / 4;
    device const uint8_t* base_data = base_buffer + gid * base_bytes_per_vec;
    uint boost_bytes_per_vec = (num_boosted + 3) / 4;
    device const uint8_t* boost_data = boost_buffer + gid * boost_bytes_per_vec;
    float b_scale = base_scales[gid];
    float b_zero  = base_zeros[gid];
    float bst_scale = boost_scales[gid];
    float bst_zero  = boost_zeros[gid];

    float partial_sum = 0.0;

    for (uint i = tid; i < dim; i += tpg) {
        // Base dequant (uniform 2-bit) — no branching
        uint base_byte = i / 4;
        uint base_offset = (i % 4) * 2;
        uint8_t base_q = (base_data[base_byte] >> base_offset) & 0x03;
        float value = float(base_q) * b_scale + b_zero;

        // O(1) boost check via prefix sum:
        // Channel i is boosted iff boost_prefix_sum[i+1] > boost_prefix_sum[i]
        uint prefix_before = boost_prefix_sum[i];
        uint prefix_after  = boost_prefix_sum[i + 1];
        if (prefix_after > prefix_before) {
            // This channel is boosted. Its index in the boost tensor = prefix_before.
            uint bst_byte = prefix_before / 4;
            uint bst_offset = (prefix_before % 4) * 2;
            uint8_t bst_q = (boost_data[bst_byte] >> bst_offset) & 0x03;
            value += float(bst_q) * bst_scale + bst_zero;
        }

        partial_sum += query_buffer[i] * value;
    }

    threadgroup float shared_sums[256];
    shared_sums[tid] = partial_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] += shared_sums[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output_buffer[gid] = shared_sums[0];
    }
}
"#;

/// Describes a Metal GPU compute context for quantized operations.
/// Platform-independent interface — actual Metal setup is done in Swift
/// via the FFI bridge, since objc2-metal has compatibility constraints.
///
/// The Rust side provides:
///   - Shader source code (MSL string above)
///   - Buffer layout specifications
///   - Dispatch geometry calculations
///
/// The Swift side handles:
///   - MTLDevice creation
///   - MTLLibrary compilation from MSL source
///   - MTLComputePipelineState creation
///   - MTLBuffer allocation with StorageModeShared
///   - Command buffer encoding and GPU dispatch
///
/// Zero-copy UMA pattern:
///   1. Swift creates MTLBuffer with .storageModeShared
///   2. Swift extracts contents() pointer (UnsafeMutableRawPointer)
///   3. Pointer + length passed to Rust via C FFI
///   4. Rust wraps as &mut [u8] slice and writes quantized data in-place
///   5. Swift dispatches compute shader — GPU reads same physical memory
///   6. Zero copies throughout
#[derive(Debug, Clone)]
pub struct MetalDispatchConfig {
    /// Kernel name in the MSL source.
    pub kernel_name: &'static str,
    /// Number of threadgroups (one per vector).
    pub threadgroups: usize,
    /// Threads per threadgroup (should be power-of-two, ≤256 for Apple GPUs).
    pub threads_per_group: usize,
}

impl MetalDispatchConfig {
    /// Compute dispatch configuration for 2-bit dequant+dot product.
    pub fn for_2bit(num_vectors: usize, dim: usize) -> Self {
        let threads = (dim / 4).min(256).max(32).next_power_of_two();
        Self {
            kernel_name: "dequantize_dot_product_2bit",
            threadgroups: num_vectors,
            threads_per_group: threads,
        }
    }

    /// Compute dispatch configuration for 4-bit dequant+dot product.
    pub fn for_4bit(num_vectors: usize, dim: usize) -> Self {
        let threads = (dim / 2).min(256).max(32).next_power_of_two();
        Self {
            kernel_name: "dequantize_dot_product_4bit",
            threadgroups: num_vectors,
            threads_per_group: threads,
        }
    }

    /// Compute dispatch configuration for Kitty two-tensor dequant+dot product.
    pub fn for_kitty(num_vectors: usize, dim: usize) -> Self {
        let threads = (dim / 4).min(256).max(32).next_power_of_two();
        Self {
            kernel_name: "dequantize_dot_product_kitty",
            threadgroups: num_vectors,
            threads_per_group: threads,
        }
    }
}

/// Build the prefix-sum buffer for Kitty boost channel indexing.
/// boost_prefix_sum[i] = number of boosted channels in [0, i).
/// Length: dim + 1 (one extra for the sentinel at the end).
/// This is computed ONCE on CPU and passed to the Metal kernel,
/// replacing the per-thread O(d) mask scan with O(1) lookups.
pub fn build_kitty_boost_prefix_sum(boost_mask: &[u8], dim: usize) -> Vec<u16> {
    let mut prefix = Vec::with_capacity(dim + 1);
    let mut count: u16 = 0;
    for i in 0..dim {
        prefix.push(count);
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        if byte_idx < boost_mask.len() && (boost_mask[byte_idx] >> bit_idx) & 1 == 1 {
            count += 1;
        }
    }
    prefix.push(count); // sentinel: total boosted channels
    prefix
}

/// Compute buffer sizes for Metal buffer allocation.
/// Returns (query_bytes, quant_bytes, scales_bytes, zeros_bytes, output_bytes).
pub fn compute_buffer_sizes_2bit(
    dim: usize,
    num_vectors: usize,
) -> (usize, usize, usize, usize, usize) {
    let query_bytes = dim * 4; // float32
    let bytes_per_vec = dim.div_ceil(4);
    let quant_bytes = bytes_per_vec * num_vectors;
    let scales_bytes = num_vectors * 4; // one float32 per vector
    let zeros_bytes = num_vectors * 4;
    let output_bytes = num_vectors * 4;
    (
        query_bytes,
        quant_bytes,
        scales_bytes,
        zeros_bytes,
        output_bytes,
    )
}

/// Compute buffer sizes for 4-bit.
pub fn compute_buffer_sizes_4bit(
    dim: usize,
    num_vectors: usize,
) -> (usize, usize, usize, usize, usize) {
    let query_bytes = dim * 4;
    let bytes_per_vec = dim.div_ceil(2);
    let quant_bytes = bytes_per_vec * num_vectors;
    let scales_bytes = num_vectors * 4;
    let zeros_bytes = num_vectors * 4;
    let output_bytes = num_vectors * 4;
    (
        query_bytes,
        quant_bytes,
        scales_bytes,
        zeros_bytes,
        output_bytes,
    )
}

/// Pack quantized vectors into a contiguous buffer suitable for Metal.
/// Returns (packed_data, scales, zeros).
pub fn pack_for_metal_2bit(
    vectors: &[&[u8]], // Each vector's packed 2-bit data
    scales: &[f32],
    zeros: &[f32],
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let total_bytes: usize = vectors.iter().map(|v| v.len()).sum();
    let mut packed = Vec::with_capacity(total_bytes);
    for v in vectors {
        packed.extend_from_slice(v);
    }
    (packed, scales.to_vec(), zeros.to_vec())
}

/// Write query vector into a Metal-compatible buffer (raw bytes).
/// The caller provides a pointer to shared GPU memory.
///
/// # Safety
/// `dst` must point to a valid allocation of at least `query.len() * 4` bytes.
/// This is used with MTLBuffer.contents() for zero-copy UMA writes.
pub unsafe fn write_query_to_metal_buffer(query: &[f32], dst: *mut u8) {
    let byte_len = std::mem::size_of_val(query);
    // SAFETY: Caller guarantees dst is valid for byte_len bytes.
    // We copy the f32 slice directly as bytes — same layout on all Apple Silicon.
    std::ptr::copy_nonoverlapping(query.as_ptr() as *const u8, dst, byte_len);
}

/// Read dot product results from a Metal output buffer.
///
/// # Safety
/// `src` must point to a valid allocation of at least `num_vectors * 4` bytes
/// containing float32 values written by the Metal compute shader.
pub unsafe fn read_scores_from_metal_buffer(src: *const u8, num_vectors: usize) -> Vec<f32> {
    let mut scores = vec![0.0_f32; num_vectors];
    // SAFETY: Caller guarantees src is valid and contains num_vectors float32s.
    std::ptr::copy_nonoverlapping(src, scores.as_mut_ptr() as *mut u8, num_vectors * 4);
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_config_2bit() {
        let config = MetalDispatchConfig::for_2bit(1000, 384);
        assert_eq!(config.kernel_name, "dequantize_dot_product_2bit");
        assert_eq!(config.threadgroups, 1000);
        assert!(config.threads_per_group.is_power_of_two());
        assert!(config.threads_per_group <= 256);
    }

    #[test]
    fn dispatch_config_4bit() {
        let config = MetalDispatchConfig::for_4bit(500, 384);
        assert_eq!(config.kernel_name, "dequantize_dot_product_4bit");
        assert_eq!(config.threadgroups, 500);
    }

    #[test]
    fn buffer_sizes_2bit() {
        let (q, data, s, z, o) = compute_buffer_sizes_2bit(384, 1000);
        assert_eq!(q, 384 * 4); // query: 1536 bytes
        assert_eq!(data, 96 * 1000); // 96 bytes per 384-dim 2-bit vector
        assert_eq!(s, 1000 * 4);
        assert_eq!(z, 1000 * 4);
        assert_eq!(o, 1000 * 4);
    }

    #[test]
    fn buffer_sizes_4bit() {
        let (q, data, s, z, o) = compute_buffer_sizes_4bit(384, 1000);
        assert_eq!(q, 384 * 4);
        assert_eq!(data, 192 * 1000); // 192 bytes per 384-dim 4-bit vector
        assert_eq!(s, 1000 * 4);
        assert_eq!(z, 1000 * 4);
        assert_eq!(o, 1000 * 4);
    }

    #[test]
    fn pack_for_metal() {
        let v1 = vec![0xAA_u8; 48]; // 384-dim at 2-bit = 96 bytes, but test with 48
        let v2 = vec![0x55_u8; 48];
        let scales = vec![0.1, 0.2];
        let zeros = vec![-1.0, -2.0];

        let (packed, s, z) = pack_for_metal_2bit(&[&v1, &v2], &scales, &zeros);
        assert_eq!(packed.len(), 96);
        assert_eq!(s, scales);
        assert_eq!(z, zeros);
    }

    #[test]
    fn metal_shader_source_compiles_as_string() {
        // Verify the MSL source is valid UTF-8 and non-empty
        assert!(METAL_SHADER_SOURCE.len() > 100);
        assert!(METAL_SHADER_SOURCE.contains("dequantize_dot_product_2bit"));
        assert!(METAL_SHADER_SOURCE.contains("dequantize_dot_product_4bit"));
        assert!(METAL_SHADER_SOURCE.contains("dequantize_dot_product_kitty"));
    }

    #[test]
    fn zero_copy_roundtrip() {
        let query = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut buffer = vec![0u8; query.len() * 4];

        // SAFETY: buffer is valid and large enough
        unsafe {
            write_query_to_metal_buffer(&query, buffer.as_mut_ptr());
        }

        // Read back
        // SAFETY: buffer contains valid float32 data
        let scores = unsafe { read_scores_from_metal_buffer(buffer.as_ptr(), query.len()) };

        for (a, b) in query.iter().zip(scores.iter()) {
            assert_eq!(*a, *b, "Zero-copy roundtrip should preserve values");
        }
    }
}
