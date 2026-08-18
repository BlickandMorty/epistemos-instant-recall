// Binary quantization: sign-bit extraction + Hamming distance.
// Compresses float32 embeddings to 1-bit per dimension (32x compression).
// 1024-dim float32 = 4096 bytes → 128 bytes binary.

/// Quantize a float32 embedding to binary (1 bit per dimension).
/// Each dimension becomes 1 if positive, 0 if negative/zero.
/// Bits are packed 8 per byte, LSB-first within each byte.
pub fn quantize_to_binary(embedding: &[f32]) -> Vec<u8> {
    let num_bytes = embedding.len().div_ceil(8);
    let mut result = Vec::with_capacity(num_bytes);

    for chunk in embedding.chunks(8) {
        let mut byte: u8 = 0;
        for (i, &value) in chunk.iter().enumerate() {
            if value > 0.0 {
                byte |= 1 << i;
            }
        }
        result.push(byte);
    }

    result
}

/// Hamming distance between two binary vectors.
/// Returns the number of differing bits (lower = more similar).
/// Uses hardware popcount (ARM: CNT instruction).
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len(), "Binary vectors must have equal length");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

/// Dot product between two float32 vectors.
/// Used in Phase 2 rescoring for precise similarity.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have equal length");
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// L2 normalize a float32 vector in-place.
pub fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_quantization_basic() {
        let embedding = vec![1.0, -2.0, 0.5, -0.1, 3.0, 0.0, -1.0, 0.7];
        let binary = quantize_to_binary(&embedding);
        assert_eq!(binary.len(), 1);
        // Expected: bits 0,2,4,7 set (positive values at indices 0,2,4,7)
        // bit0=1, bit1=0, bit2=1, bit3=0, bit4=1, bit5=0, bit6=0, bit7=1
        assert_eq!(binary[0], 0b10010101);
    }

    #[test]
    fn binary_quantization_multi_byte() {
        let embedding = vec![1.0; 16]; // 16 dims → 2 bytes, all positive
        let binary = quantize_to_binary(&embedding);
        assert_eq!(binary.len(), 2);
        assert_eq!(binary[0], 0xFF);
        assert_eq!(binary[1], 0xFF);
    }

    #[test]
    fn binary_quantization_partial_byte() {
        let embedding = vec![1.0, -1.0, 1.0]; // 3 dims → 1 byte
        let binary = quantize_to_binary(&embedding);
        assert_eq!(binary.len(), 1);
        // bit0=1, bit1=0, bit2=1
        assert_eq!(binary[0], 0b00000101);
    }

    #[test]
    fn hamming_distance_identical() {
        let a = vec![0xFF, 0x00, 0xAA];
        let b = vec![0xFF, 0x00, 0xAA];
        assert_eq!(hamming_distance(&a, &b), 0);
    }

    #[test]
    fn hamming_distance_opposite() {
        let a = vec![0xFF];
        let b = vec![0x00];
        assert_eq!(hamming_distance(&a, &b), 8);
    }

    #[test]
    fn hamming_distance_one_bit() {
        let a = vec![0b00000001];
        let b = vec![0b00000000];
        assert_eq!(hamming_distance(&a, &b), 1);
    }

    #[test]
    fn dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[test]
    fn dot_product_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((dot_product(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn normalize_basic() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn normalize_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        normalize(&mut v);
        // Should not panic, stays at zero
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn binary_quantization_roundtrip_preserves_sign() {
        let embedding: Vec<f32> = (0..1024)
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        let binary = quantize_to_binary(&embedding);
        assert_eq!(binary.len(), 128); // 1024 / 8

        // Verify each bit matches the sign
        for (i, &val) in embedding.iter().enumerate() {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let bit_set = (binary[byte_idx] >> bit_idx) & 1 == 1;
            assert_eq!(bit_set, val > 0.0, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn hamming_distance_large_vectors() {
        // 1024-dim → 128 bytes
        let a = vec![0xAA; 128];
        let b = vec![0x55; 128];
        // AA = 10101010, 55 = 01010101 → XOR = FF → 8 bits per byte
        assert_eq!(hamming_distance(&a, &b), 128 * 8);
    }
}
