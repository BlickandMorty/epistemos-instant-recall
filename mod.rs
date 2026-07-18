// Ω18 — Instant Recall: Binary-quantized vector index with two-phase retrieval.
// Sub-3ms vault-wide semantic search for note recall as you type.
//
// Architecture:
//   Text → Trigram Embedder → float32 vector
//        → Binary Quantizer → 1-bit signature (sign-bit extraction)
//        → Flat Binary Index → Hamming distance scan (ARM NEON popcount)
//        → Float32 Rescorer → dot product on top-K candidates
//        → Top-5 relevant notes returned to Swift
//
// The flat binary scan is optimal for vaults up to ~500K notes:
//   128 bytes/note × 500K = 64MB, scanned at ~350 GB/s = 0.18ms.
// HNSW is deferred until vault sizes exceed this threshold.

pub mod butterfly;
pub mod embedder;
pub mod fusion;
pub mod index;
pub mod kitty;
pub mod kv_cache_quant;
pub mod metal_quant;
pub mod progressive;
pub mod quantizer;
pub mod segment;
pub mod tmac;
pub mod turbo_quant;

pub use butterfly::ButterflyRotation;
pub use embedder::TrigramEmbedder;
pub use fusion::{
    CrossEncoder, FusedResult, FusionConfig, HybridSearchPipeline, RetrievalHit, RetrievalSource,
};
pub use index::{InstantRecallIndex, RecallResult};
pub use kitty::{KittyBoostMap, KittyConfig, KittyVector};
pub use kv_cache_quant::{KVPrecision, KVTunerProfile, ProgressiveKVCache};
pub use quantizer::{hamming_distance, quantize_to_binary};
pub use segment::{SegmentConfig, SegmentSearchResult, SegmentedIndex};
pub use tmac::TMacVector;
pub use turbo_quant::{TurboQuantBits, TurboQuantVector};

/// Configuration for the instant recall system.
#[derive(Debug, Clone)]
pub struct InstantRecallConfig {
    /// Embedding dimension (default: 1024).
    pub dimension: usize,
    /// Number of candidates from binary scan (Phase 1). Default: 100.
    pub binary_top_k: usize,
    /// Number of final results after float32 rescoring (Phase 2). Default: 5.
    pub final_top_k: usize,
}

impl Default for InstantRecallConfig {
    fn default() -> Self {
        Self {
            dimension: 1024,
            binary_top_k: 100,
            final_top_k: 5,
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn end_to_end_index_and_search() {
        let config = InstantRecallConfig::default();
        let embedder = TrigramEmbedder::new(config.dimension);
        let mut index = InstantRecallIndex::new(config.clone());

        // Index some notes
        let notes = vec![
            ("note-1", "Rust programming language systems design"),
            ("note-2", "Swift macOS native application development"),
            ("note-3", "Machine learning neural networks deep learning"),
            ("note-4", "Cooking recipes Italian pasta carbonara"),
            ("note-5", "Quantum physics wave function collapse"),
        ];

        for (id, text) in &notes {
            let embedding = embedder.encode(text);
            index.insert(id.to_string(), embedding, text.to_string());
        }

        assert_eq!(index.len(), 5);

        // Search for something related to programming
        let query_embedding = embedder.encode("systems programming with Rust");
        let results = index.search(&query_embedding, 3);

        assert!(!results.is_empty());
        assert!(results.len() <= 3);
        // The Rust note should be the top result
        assert_eq!(results[0].doc_id, "note-1");
    }

    #[test]
    fn empty_index_returns_empty_results() {
        let config = InstantRecallConfig::default();
        let index = InstantRecallIndex::new(config);
        let query = vec![0.0f32; 1024];
        let results = index.search(&query, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn search_with_top_k_larger_than_index() {
        let config = InstantRecallConfig::default();
        let embedder = TrigramEmbedder::new(config.dimension);
        let mut index = InstantRecallIndex::new(config);

        let embedding = embedder.encode("hello world");
        index.insert("only-one".into(), embedding, "hello world".into());

        let query = embedder.encode("hello");
        let results = index.search(&query, 100);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn remove_document_from_index() {
        let config = InstantRecallConfig::default();
        let embedder = TrigramEmbedder::new(config.dimension);
        let mut index = InstantRecallIndex::new(config);

        let emb = embedder.encode("test document");
        index.insert("doc-1".into(), emb.clone(), "test document".into());
        assert_eq!(index.len(), 1);

        index.remove("doc-1");
        assert_eq!(index.len(), 0);

        let results = index.search(&emb, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn duplicate_id_replaces_existing() {
        let config = InstantRecallConfig::default();
        let embedder = TrigramEmbedder::new(config.dimension);
        let mut index = InstantRecallIndex::new(config);

        let emb1 = embedder.encode("original text");
        index.insert("doc-1".into(), emb1, "original text".into());

        let emb2 = embedder.encode("updated text");
        index.insert("doc-1".into(), emb2, "updated text".into());

        assert_eq!(index.len(), 1);

        let query = embedder.encode("updated");
        let results = index.search(&query, 1);
        assert_eq!(results[0].doc_id, "doc-1");
        assert_eq!(results[0].text, "updated text");
    }

    #[test]
    fn latency_under_threshold_for_1k_notes() {
        let config = InstantRecallConfig::default();
        let embedder = TrigramEmbedder::new(config.dimension);
        let mut index = InstantRecallIndex::new(config);

        // Insert 1000 notes
        for i in 0..1000 {
            let text = format!(
                "Note number {} about topic {} with content {}",
                i,
                i % 50,
                i * 7
            );
            let embedding = embedder.encode(&text);
            index.insert(format!("note-{}", i), embedding, text);
        }

        // Measure search latency
        let query = embedder.encode("topic 25 content");
        let start = std::time::Instant::now();
        let _results = index.search(&query, 5);
        let elapsed = start.elapsed();

        // Must be under 10ms for 1K notes (generous margin over <3ms target)
        assert!(
            elapsed.as_millis() < 10,
            "Search took {}ms, expected <10ms",
            elapsed.as_millis()
        );
    }
}
