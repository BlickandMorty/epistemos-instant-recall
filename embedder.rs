// Trigram embedder: fast, deterministic text-to-vector encoding.
// Placeholder for model2vec (which requires weight files).
//
// Approach: Hash character trigrams into a fixed-dimension vector,
// then L2-normalize. This produces surprisingly good embeddings for
// semantic similarity when the vocabulary is consistent (e.g., notes).
//
// When model2vec-rs becomes available, swap this implementation
// behind the same encode() interface.

use crate::instant_recall::quantizer;

/// Fast trigram-based text embedder.
/// Deterministic: same text always produces the same embedding.
pub struct TrigramEmbedder {
    dimension: usize,
}

impl TrigramEmbedder {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Encode text to a normalized float32 embedding vector.
    /// Uses character trigram hashing with dimension folding.
    pub fn encode(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.dimension];

        let lower = text.to_lowercase();
        let chars: Vec<char> = lower.chars().collect();

        if chars.len() < 3 {
            // For very short text, use unigram hashing
            for (i, &ch) in chars.iter().enumerate() {
                let hash = Self::hash_char(ch, i as u64);
                let idx = hash as usize % self.dimension;
                embedding[idx] += 1.0;
            }
        } else {
            // Trigram hashing: each trigram contributes to multiple dimensions
            for window in chars.windows(3) {
                let trigram_hash = Self::hash_trigram(window[0], window[1], window[2]);

                // Primary dimension
                let idx1 = trigram_hash as usize % self.dimension;
                embedding[idx1] += 1.0;

                // Secondary dimension (cross-feature interaction)
                let idx2 = (trigram_hash.wrapping_mul(2654435761)) as usize % self.dimension;
                embedding[idx2] += 0.5;
            }

            // Word-level features for semantic separation
            for word in lower.split_whitespace() {
                let word_hash = Self::hash_str(word);
                let idx = word_hash as usize % self.dimension;
                embedding[idx] += 2.0; // Words weighted higher than trigrams
            }
        }

        quantizer::normalize(&mut embedding);
        embedding
    }

    #[inline]
    fn hash_trigram(a: char, b: char, c: char) -> u64 {
        let mut h: u64 = 14695981039346656037; // FNV-1a offset basis
        h ^= a as u64;
        h = h.wrapping_mul(1099511628211); // FNV prime
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
        h ^= c as u64;
        h = h.wrapping_mul(1099511628211);
        h
    }

    #[inline]
    fn hash_char(ch: char, salt: u64) -> u64 {
        let mut h: u64 = 14695981039346656037;
        h ^= ch as u64;
        h = h.wrapping_mul(1099511628211);
        h ^= salt;
        h = h.wrapping_mul(1099511628211);
        h
    }

    #[inline]
    fn hash_str(s: &str) -> u64 {
        let mut h: u64 = 14695981039346656037;
        for byte in s.bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(1099511628211);
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_produces_correct_dimension() {
        let embedder = TrigramEmbedder::new(1024);
        let embedding = embedder.encode("hello world");
        assert_eq!(embedding.len(), 1024);
    }

    #[test]
    fn encode_is_normalized() {
        let embedder = TrigramEmbedder::new(1024);
        let embedding = embedder.encode("hello world");
        let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Expected unit norm, got {}",
            norm
        );
    }

    #[test]
    fn encode_is_deterministic() {
        let embedder = TrigramEmbedder::new(1024);
        let a = embedder.encode("test document about Rust");
        let b = embedder.encode("test document about Rust");
        assert_eq!(a, b);
    }

    #[test]
    fn similar_texts_have_high_similarity() {
        let embedder = TrigramEmbedder::new(1024);
        let a = embedder.encode("Rust programming language");
        let b = embedder.encode("Rust programming systems");

        let sim: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        assert!(sim > 0.5, "Expected high similarity, got {}", sim);
    }

    #[test]
    fn dissimilar_texts_have_low_similarity() {
        let embedder = TrigramEmbedder::new(1024);
        let a = embedder.encode("Rust programming language systems design");
        let b = embedder.encode("Italian cooking pasta recipes carbonara");

        let sim: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        assert!(sim < 0.3, "Expected low similarity, got {}", sim);
    }

    #[test]
    fn empty_text_produces_zero_vector() {
        let embedder = TrigramEmbedder::new(1024);
        let embedding = embedder.encode("");
        // All zeros (can't normalize a zero vector)
        assert!(embedding.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn short_text_still_embeds() {
        let embedder = TrigramEmbedder::new(1024);
        let embedding = embedder.encode("hi");
        // Should produce non-zero vector despite being shorter than trigram
        let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            norm > 0.9,
            "Expected near-unit norm for short text, got {}",
            norm
        );
    }

    #[test]
    fn unicode_text_works() {
        let embedder = TrigramEmbedder::new(1024);
        let embedding = embedder.encode("日本語のテスト");
        let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn case_insensitive() {
        let embedder = TrigramEmbedder::new(1024);
        let a = embedder.encode("Hello World");
        let b = embedder.encode("hello world");
        assert_eq!(a, b);
    }
}
