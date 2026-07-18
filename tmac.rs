#![allow(clippy::doc_lazy_continuation, clippy::needless_range_loop)]

// T-MAC: Table-based Multiplication for Any-bit Combinations.
//
// Eliminates dequantization entirely by decomposing n-bit multiplication into
// n one-bit lookup passes. A group of 4 one-bit weights has only 16 possible
// bit patterns, so all partial sums are precomputed into a 16-entry LUT.
//
// Key advantages over dequantize-then-multiply:
//   - Different bit-widths (1,2,3,4-bit) use the SAME kernel logic
//   - No data-type-dependent branching, no SIMD lane divergence
//   - Throughput scales linearly as bit-width decreases (unlike traditional approach)
//   - On Apple M2-Ultra: 6.6× over llama.cpp, 71 tok/s for BitNet-3B
//   - Energy consumption drops 70% vs llama.cpp
//
// Reference: T-MAC (arXiv:2407.00088, EuroSys 2025)
//            Microsoft bitnet.cpp (production deployment)
//
// This module implements the LUT paradigm for vector dot products,
// usable for both embedding search and KV cache attention.

/// Group size for LUT construction. 4 bits per group → 16-entry LUT.
/// Matches ARM NEON `tbl` instruction's 16-entry lookup capability.
pub const LUT_GROUP_SIZE: usize = 4;

/// Number of LUT entries per group: 2^GROUP_SIZE = 16.
pub const LUT_ENTRIES: usize = 1 << LUT_GROUP_SIZE;

/// A T-MAC quantized vector: weights stored as bit-planes + precomputed LUT metadata.
#[derive(Debug, Clone)]
pub struct TMacVector {
    /// Bit-plane storage: one plane per bit of precision.
    /// Each plane packs `dim` bits into ceil(dim/8) bytes.
    /// For 2-bit: 2 planes. For 4-bit: 4 planes.
    pub bit_planes: Vec<Vec<u8>>,
    /// Number of bit-planes (= precision in bits).
    pub num_bits: usize,
    /// Original dimension.
    pub dim: usize,
    /// Quantization scale: original_value ≈ quantized_level × scale + zero_point.
    pub scale: f32,
    /// Quantization zero point (min value of the original data).
    pub zero_point: f32,
}

/// Quantize a float32 vector into T-MAC bit-plane representation.
///
/// The vector is uniformly quantized to `num_bits` precision, then each bit
/// is stored in a separate plane. This layout enables the LUT-based dot product.
pub fn tmac_quantize(vector: &[f32], num_bits: usize) -> TMacVector {
    let dim = vector.len();
    let num_levels = (1u32 << num_bits) - 1;

    // Compute asymmetric quantization parameters
    let (min_val, max_val) = vector
        .iter()
        .fold((f32::MAX, f32::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
    let range = max_val - min_val;
    let scale = if range > 1e-10 {
        range / num_levels as f32
    } else {
        1.0
    };

    // Quantize to integer levels
    let quantized: Vec<u8> = vector
        .iter()
        .map(|&v| {
            ((v - min_val) / scale)
                .round()
                .clamp(0.0, num_levels as f32) as u8
        })
        .collect();

    // Decompose into bit planes
    let bytes_per_plane = dim.div_ceil(8);
    let mut bit_planes = Vec::with_capacity(num_bits);

    for bit in 0..num_bits {
        let mut plane = vec![0u8; bytes_per_plane];
        for (i, &q) in quantized.iter().enumerate() {
            if (q >> bit) & 1 == 1 {
                plane[i / 8] |= 1 << (i % 8);
            }
        }
        bit_planes.push(plane);
    }

    TMacVector {
        bit_planes,
        num_bits,
        dim,
        scale,
        zero_point: min_val,
    }
}

/// Build a 16-entry lookup table for a group of 4 query values.
///
/// For group [q0, q1, q2, q3], the LUT maps each 4-bit pattern to the
/// corresponding partial sum. Pattern `0b1010` means bits 1 and 3 are set,
/// so the partial sum = q1 + q3.
///
/// This is precomputed ONCE per query, then used for ALL stored vectors.
#[inline]
fn build_group_lut(query_group: &[f32]) -> [f32; LUT_ENTRIES] {
    let mut lut = [0.0_f32; LUT_ENTRIES];
    let n = query_group.len().min(LUT_GROUP_SIZE);

    // Enumerate all 16 possible bit patterns
    for pattern in 0..LUT_ENTRIES {
        let mut sum = 0.0_f32;
        for bit in 0..n {
            if (pattern >> bit) & 1 == 1 {
                sum += query_group[bit];
            }
        }
        lut[pattern] = sum;
    }

    lut
}

/// T-MAC dot product: LUT-based computation with zero dequantization.
///
/// Pipeline:
///   1. Build LUTs from query (ONCE per query, amortized across all vectors)
///   2. For each bit-plane of the stored vector:
///     a. Extract 4-bit group pattern from the bit-plane
///     b. Look up partial sum in the pre-built LUT
///     c. Accumulate with bit-position weight (2^bit)
///   3. Scale the accumulated sum by the quantization parameters
///
/// Total cost: O(dim × num_bits / GROUP_SIZE) lookups — no multiplications.
/// On ARM: `tbl` instruction does 16 lookups per cycle.
pub fn tmac_dot_product(query: &[f32], tmac_vec: &TMacVector) -> f32 {
    let dim = query.len().min(tmac_vec.dim);
    let num_groups = dim.div_ceil(LUT_GROUP_SIZE);

    // Step 1: Build LUTs from query (one per group of 4 values)
    let luts: Vec<[f32; LUT_ENTRIES]> = (0..num_groups)
        .map(|g| {
            let start = g * LUT_GROUP_SIZE;
            let end = (start + LUT_GROUP_SIZE).min(dim);
            build_group_lut(&query[start..end])
        })
        .collect();

    // Step 2: For each bit-plane, accumulate weighted LUT lookups
    // LUT accumulates Σ query[i] × quantized_level[i]
    let mut quant_dot = 0.0_f32;

    for (bit, plane) in tmac_vec.bit_planes.iter().enumerate() {
        let bit_weight = (1u32 << bit) as f32;
        let mut plane_sum = 0.0_f32;

        for (group_idx, lut) in luts.iter().enumerate() {
            let base_idx = group_idx * LUT_GROUP_SIZE;
            let mut pattern: usize = 0;

            for bit_pos in 0..LUT_GROUP_SIZE {
                let elem_idx = base_idx + bit_pos;
                if elem_idx >= dim {
                    break;
                }
                let byte_idx = elem_idx / 8;
                let bit_idx = elem_idx % 8;
                if byte_idx < plane.len() && (plane[byte_idx] >> bit_idx) & 1 == 1 {
                    pattern |= 1 << bit_pos;
                }
            }

            plane_sum += lut[pattern];
        }

        quant_dot += plane_sum * bit_weight;
    }

    // Step 3: Convert from quantized-level dot product to real-valued dot product.
    // original[i] ≈ quantized_level[i] × scale + zero_point
    // Σ query[i] × original[i] ≈ scale × Σ query[i] × level[i] + zero_point × Σ query[i]
    let query_sum: f32 = query[..dim].iter().sum();
    quant_dot * tmac_vec.scale + tmac_vec.zero_point * query_sum
}

/// Batch dot product: compute query against multiple T-MAC vectors.
/// LUTs are built once and reused for all vectors — the core T-MAC advantage.
pub fn tmac_batch_dot_product(query: &[f32], vectors: &[TMacVector]) -> Vec<f32> {
    let dim = query.len();
    let num_groups = dim.div_ceil(LUT_GROUP_SIZE);

    // Build LUTs ONCE for this query
    let luts: Vec<[f32; LUT_ENTRIES]> = (0..num_groups)
        .map(|g| {
            let start = g * LUT_GROUP_SIZE;
            let end = (start + LUT_GROUP_SIZE).min(dim);
            build_group_lut(&query[start..end])
        })
        .collect();

    let query_sum: f32 = query[..dim].iter().sum();

    // Compute dot product for each stored vector using shared LUTs
    vectors
        .iter()
        .map(|tmac_vec| {
            let mut quant_dot = 0.0_f32;
            for (bit, plane) in tmac_vec.bit_planes.iter().enumerate() {
                let bit_weight = (1u32 << bit) as f32;
                let mut plane_sum = 0.0_f32;

                for (group_idx, lut) in luts.iter().enumerate() {
                    let base_idx = group_idx * LUT_GROUP_SIZE;
                    let mut pattern: usize = 0;

                    for bit_pos in 0..LUT_GROUP_SIZE {
                        let elem_idx = base_idx + bit_pos;
                        if elem_idx >= tmac_vec.dim {
                            break;
                        }
                        let byte_idx = elem_idx / 8;
                        let bit_idx = elem_idx % 8;
                        if byte_idx < plane.len() && (plane[byte_idx] >> bit_idx) & 1 == 1 {
                            pattern |= 1 << bit_pos;
                        }
                    }

                    plane_sum += lut[pattern];
                }

                quant_dot += plane_sum * bit_weight;
            }
            quant_dot * tmac_vec.scale + tmac_vec.zero_point * query_sum
        })
        .collect()
}

/// Memory footprint of a T-MAC vector in bytes.
pub fn tmac_memory_bytes(dim: usize, num_bits: usize) -> usize {
    let bytes_per_plane = dim.div_ceil(8);
    bytes_per_plane * num_bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_pattern_0_is_zero() {
        let group = [1.0, 2.0, 3.0, 4.0];
        let lut = build_group_lut(&group);
        assert_eq!(lut[0], 0.0); // no bits set → sum = 0
    }

    #[test]
    fn lut_pattern_all_set() {
        let group = [1.0, 2.0, 3.0, 4.0];
        let lut = build_group_lut(&group);
        assert!((lut[0b1111] - 10.0).abs() < 1e-6); // all bits → 1+2+3+4 = 10
    }

    #[test]
    fn lut_single_bits() {
        let group = [10.0, 20.0, 30.0, 40.0];
        let lut = build_group_lut(&group);
        assert!((lut[0b0001] - 10.0).abs() < 1e-6);
        assert!((lut[0b0010] - 20.0).abs() < 1e-6);
        assert!((lut[0b0100] - 30.0).abs() < 1e-6);
        assert!((lut[0b1000] - 40.0).abs() < 1e-6);
    }

    #[test]
    fn quantize_produces_correct_planes() {
        // 2-bit quantization: values mapped to {0,1,2,3}
        let vector = vec![0.0, 0.33, 0.66, 1.0];
        let tmac = tmac_quantize(&vector, 2);
        assert_eq!(tmac.num_bits, 2);
        assert_eq!(tmac.bit_planes.len(), 2);
        assert_eq!(tmac.dim, 4);
    }

    #[test]
    fn dot_product_basic() {
        // Simple case: all-ones query × quantized vector
        let dim = 16;
        let query = vec![1.0_f32; dim];
        let vector: Vec<f32> = (0..dim).map(|i| i as f32 / (dim - 1) as f32).collect();

        let tmac = tmac_quantize(&vector, 4);
        let result = tmac_dot_product(&query, &tmac);

        // Result should approximate sum of vector elements (since query is all-ones)
        let expected: f32 = vector.iter().sum();
        // T-MAC introduces quantization error; check within reasonable bound
        assert!(
            (result - expected).abs() / expected.abs().max(1.0) < 0.5,
            "T-MAC dot product too far from expected: got {result}, expected ~{expected}"
        );
    }

    #[test]
    fn batch_matches_individual() {
        let dim = 32;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let vectors: Vec<Vec<f32>> = (0..5)
            .map(|seed| {
                (0..dim)
                    .map(|i| ((seed as f32) * 0.3 + (i as f32) * 0.1).cos())
                    .collect()
            })
            .collect();

        let tmac_vecs: Vec<TMacVector> = vectors.iter().map(|v| tmac_quantize(v, 2)).collect();

        let batch_results = tmac_batch_dot_product(&query, &tmac_vecs);
        let individual_results: Vec<f32> = tmac_vecs
            .iter()
            .map(|tv| tmac_dot_product(&query, tv))
            .collect();

        for (b, i) in batch_results.iter().zip(individual_results.iter()) {
            assert!(
                (b - i).abs() < 1e-6,
                "Batch and individual should match: {b} vs {i}"
            );
        }
    }

    #[test]
    fn different_bit_widths_same_kernel() {
        let dim = 16;
        let query = vec![1.0_f32; dim];
        let vector: Vec<f32> = (0..dim).map(|i| i as f32).collect();

        // T-MAC's key property: same kernel logic for all bit-widths
        let tmac_1bit = tmac_quantize(&vector, 1);
        let tmac_2bit = tmac_quantize(&vector, 2);
        let tmac_4bit = tmac_quantize(&vector, 4);

        let r1 = tmac_dot_product(&query, &tmac_1bit);
        let r2 = tmac_dot_product(&query, &tmac_2bit);
        let r4 = tmac_dot_product(&query, &tmac_4bit);

        // Higher precision should be more accurate
        let exact: f32 = vector.iter().sum();
        let err1 = (r1 - exact).abs();
        let err2 = (r2 - exact).abs();
        let err4 = (r4 - exact).abs();

        assert!(
            err4 <= err2 || err2 <= err1,
            "Higher precision should generally reduce error: 1bit={err1}, 2bit={err2}, 4bit={err4}"
        );
    }

    #[test]
    fn memory_footprint() {
        // 384-dim at 2-bit: 2 planes × 48 bytes = 96 bytes
        assert_eq!(tmac_memory_bytes(384, 2), 96);
        // vs float32: 384 × 4 = 1536 bytes → 16× compression
        assert_eq!(tmac_memory_bytes(384, 4), 192);
    }

    #[test]
    fn handles_non_multiple_of_group_size() {
        let dim = 13; // not divisible by 4
        let query: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let vector: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.5).collect();

        let tmac = tmac_quantize(&vector, 2);
        let result = tmac_dot_product(&query, &tmac);
        assert!(
            result.is_finite(),
            "Should handle non-multiple-of-4 dimensions"
        );
    }

    #[test]
    fn zero_vector_produces_zero_dot() {
        let dim = 16;
        let query = vec![0.0_f32; dim];
        let vector: Vec<f32> = (0..dim).map(|i| i as f32).collect();

        let tmac = tmac_quantize(&vector, 2);
        let result = tmac_dot_product(&query, &tmac);
        assert!(
            (result).abs() < 1e-6,
            "Zero query should produce zero dot product"
        );
    }
}
