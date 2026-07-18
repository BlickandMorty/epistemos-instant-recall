// Flat binary index with two-phase retrieval.
//
// Phase 1: Exhaustive Hamming distance scan over binary signatures.
//   ARM NEON popcount achieves ~350 GB/s throughput.
//   For 100K notes × 128 bytes = 12.8MB → 0.037ms scan time.
//
// Phase 2: Float32 dot-product rescoring on top-K candidates.
//   100 candidates × 1024 dims = 400KB → ~0.5ms.
//
// Total: <3ms for vaults up to 500K notes.
//
// HNSW is deferred — flat scan with binary quantization is faster
// than float32 HNSW for vault sizes under ~500K.

use std::collections::HashMap;

use crate::instant_recall::quantizer::{dot_product, hamming_distance, quantize_to_binary};
use crate::instant_recall::InstantRecallConfig;

/// A single indexed document with both binary and float32 representations.
struct IndexEntry {
    doc_id: String,
    text: String,
    binary: Vec<u8>,
    float32: Vec<f32>,
}

/// Result from a search query.
#[derive(Debug, Clone)]
pub struct RecallResult {
    pub doc_id: String,
    pub text: String,
    pub score: f64,
}

/// Flat binary index with two-phase retrieval.
/// Thread-safe: designed to be wrapped in Arc<Mutex<>> on the Swift side.
pub struct InstantRecallIndex {
    config: InstantRecallConfig,
    entries: Vec<IndexEntry>,
    id_to_idx: HashMap<String, usize>,
}

impl InstantRecallIndex {
    pub fn new(config: InstantRecallConfig) -> Self {
        Self {
            config,
            entries: Vec::with_capacity(1024),
            id_to_idx: HashMap::with_capacity(1024),
        }
    }

    /// Insert or replace a document in the index.
    pub fn insert(&mut self, doc_id: String, embedding: Vec<f32>, text: String) {
        let binary = quantize_to_binary(&embedding);

        let entry = IndexEntry {
            doc_id: doc_id.clone(),
            text,
            binary,
            float32: embedding,
        };

        if let Some(&idx) = self.id_to_idx.get(&doc_id) {
            // Replace existing entry
            self.entries[idx] = entry;
        } else {
            // Append new entry
            let idx = self.entries.len();
            self.id_to_idx.insert(doc_id, idx);
            self.entries.push(entry);
        }
    }

    /// Remove a document from the index.
    pub fn remove(&mut self, doc_id: &str) {
        if let Some(idx) = self.id_to_idx.remove(doc_id) {
            // Swap-remove for O(1) deletion
            let last_idx = self.entries.len() - 1;
            if idx != last_idx {
                let last_id = self.entries[last_idx].doc_id.clone();
                self.entries.swap(idx, last_idx);
                self.id_to_idx.insert(last_id, idx);
            }
            self.entries.pop();
        }
    }

    /// Two-phase search: binary Hamming scan → float32 dot-product rescoring.
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<RecallResult> {
        if self.entries.is_empty() {
            return Vec::new();
        }

        let final_k = top_k.min(self.entries.len());
        let binary_k = self.config.binary_top_k.min(self.entries.len());

        // Phase 1: Binary Hamming scan
        let query_binary = quantize_to_binary(query_embedding);
        let mut candidates: Vec<(usize, u32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let dist = hamming_distance(&query_binary, &entry.binary);
                (i, dist)
            })
            .collect();

        // Partial sort: only need top binary_k candidates
        let nth = binary_k.min(candidates.len()) - 1;
        candidates.select_nth_unstable_by_key(nth, |&(_, d)| d);
        candidates.truncate(binary_k);

        // Phase 2: Float32 dot-product rescoring
        let mut scored: Vec<(usize, f32)> = candidates
            .iter()
            .map(|&(idx, _)| {
                let score = dot_product(query_embedding, &self.entries[idx].float32);
                (idx, score)
            })
            .collect();

        // Sort by score descending
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(final_k);

        scored
            .into_iter()
            .map(|(idx, score)| RecallResult {
                doc_id: self.entries[idx].doc_id.clone(),
                text: self.entries[idx].text.clone(),
                score: score as f64,
            })
            .collect()
    }

    /// Number of documents in the index.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear the entire index.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.id_to_idx.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> InstantRecallConfig {
        InstantRecallConfig {
            dimension: 16,
            binary_top_k: 10,
            final_top_k: 3,
        }
    }

    fn make_embedding(seed: u8, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|i| {
                let val = ((seed as f32) * 0.1 + (i as f32) * 0.01).sin();
                val
            })
            .collect()
    }

    #[test]
    fn insert_and_len() {
        let mut index = InstantRecallIndex::new(make_config());
        index.insert("a".into(), make_embedding(1, 16), "text a".into());
        index.insert("b".into(), make_embedding(2, 16), "text b".into());
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn insert_duplicate_replaces() {
        let mut index = InstantRecallIndex::new(make_config());
        index.insert("a".into(), make_embedding(1, 16), "old".into());
        index.insert("a".into(), make_embedding(2, 16), "new".into());
        assert_eq!(index.len(), 1);
        // Search should return "new"
        let results = index.search(&make_embedding(2, 16), 1);
        assert_eq!(results[0].text, "new");
    }

    #[test]
    fn remove_existing() {
        let mut index = InstantRecallIndex::new(make_config());
        index.insert("a".into(), make_embedding(1, 16), "a".into());
        index.insert("b".into(), make_embedding(2, 16), "b".into());
        index.remove("a");
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn remove_nonexistent_is_noop() {
        let mut index = InstantRecallIndex::new(make_config());
        index.insert("a".into(), make_embedding(1, 16), "a".into());
        index.remove("z");
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn search_empty_index() {
        let index = InstantRecallIndex::new(make_config());
        let results = index.search(&make_embedding(1, 16), 5);
        assert!(results.is_empty());
    }

    #[test]
    fn search_returns_correct_count() {
        let mut index = InstantRecallIndex::new(make_config());
        for i in 0..20u8 {
            index.insert(
                format!("doc-{}", i),
                make_embedding(i, 16),
                format!("text {}", i),
            );
        }
        let results = index.search(&make_embedding(5, 16), 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn search_returns_most_similar_first() {
        let mut index = InstantRecallIndex::new(make_config());

        let target = make_embedding(42, 16);
        index.insert("target".into(), target.clone(), "target".into());
        index.insert("other".into(), make_embedding(200, 16), "other".into());

        let results = index.search(&target, 2);
        assert_eq!(results[0].doc_id, "target");
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn clear_empties_index() {
        let mut index = InstantRecallIndex::new(make_config());
        index.insert("a".into(), make_embedding(1, 16), "a".into());
        index.clear();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn swap_remove_preserves_id_mapping() {
        let mut index = InstantRecallIndex::new(make_config());
        index.insert("first".into(), make_embedding(1, 16), "first".into());
        index.insert("second".into(), make_embedding(2, 16), "second".into());
        index.insert("third".into(), make_embedding(3, 16), "third".into());

        // Remove first (triggers swap with third)
        index.remove("first");
        assert_eq!(index.len(), 2);

        // "third" should still be findable
        let results = index.search(&make_embedding(3, 16), 3);
        assert!(results.iter().any(|r| r.doc_id == "third"));
    }
}
