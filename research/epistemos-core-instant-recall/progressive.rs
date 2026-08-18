// 3-layer progressive memory retrieval.
//
// Layer 1 (Working): current session context — recent messages, active tools.
//   Always in-context, O(1) lookup, no search needed. Highest priority.
//
// Layer 2 (Session): recent sessions from the same task/day.
//   Searched by semantic similarity via instant recall index.
//   Results boosted by recency (decay factor).
//
// Layer 3 (Long-term): vault-wide knowledge.
//   Full instant recall search across all indexed notes.
//   Lower priority, broader scope.
//
// The ProgressiveRetriever merges results from all 3 layers with weighted scoring.

use crate::instant_recall::embedder::TrigramEmbedder;
use crate::instant_recall::{InstantRecallConfig, InstantRecallIndex};

/// A memory entry with its source layer and metadata.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub doc_id: String,
    pub text: String,
    pub score: f64,
    pub layer: MemoryLayer,
    /// Seconds since this entry was created/modified. Used for recency decay.
    pub age_seconds: u64,
}

/// Which memory layer an entry came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayer {
    Working,
    Session,
    LongTerm,
}

impl MemoryLayer {
    /// Base weight multiplier for this layer.
    fn weight(&self) -> f64 {
        match self {
            Self::Working => 3.0,
            Self::Session => 2.0,
            Self::LongTerm => 1.0,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Working => "working",
            Self::Session => "session",
            Self::LongTerm => "long_term",
        }
    }
}

/// Configuration for progressive retrieval.
#[derive(Debug, Clone)]
pub struct ProgressiveConfig {
    /// Max results from working memory (Layer 1).
    pub working_limit: usize,
    /// Max results from session memory (Layer 2).
    pub session_limit: usize,
    /// Max results from long-term memory (Layer 3).
    pub longterm_limit: usize,
    /// Total max results returned after merging.
    pub total_limit: usize,
    /// Minimum score threshold (entries below this are dropped).
    pub min_score: f64,
    /// Recency half-life in seconds. Entries older than this get halved weight.
    pub recency_half_life_secs: u64,
}

impl Default for ProgressiveConfig {
    fn default() -> Self {
        Self {
            working_limit: 5,
            session_limit: 10,
            longterm_limit: 15,
            total_limit: 20,
            min_score: 0.1,
            recency_half_life_secs: 3600, // 1 hour
        }
    }
}

/// 3-layer progressive memory retriever.
pub struct ProgressiveRetriever {
    config: ProgressiveConfig,
    /// Layer 1: working memory items (manually managed, always in scope).
    working_items: Vec<WorkingItem>,
    /// Layer 2: session memory index (recent sessions).
    session_index: InstantRecallIndex,
    /// Layer 3: long-term memory index (vault-wide).
    longterm_index: InstantRecallIndex,
    /// Shared embedder for query encoding.
    embedder: TrigramEmbedder,
}

/// A working memory item (always in-context).
#[derive(Debug, Clone)]
struct WorkingItem {
    doc_id: String,
    text: String,
    keywords: Vec<String>,
    age_seconds: u64,
}

impl ProgressiveRetriever {
    /// Create a new progressive retriever.
    pub fn new(config: ProgressiveConfig) -> Self {
        let recall_config = InstantRecallConfig::default();
        Self {
            config,
            working_items: Vec::with_capacity(16),
            session_index: InstantRecallIndex::new(recall_config.clone()),
            longterm_index: InstantRecallIndex::new(recall_config),
            embedder: TrigramEmbedder::new(1024),
        }
    }

    // ── Layer 1: Working Memory ──

    /// Add an item to working memory.
    pub fn add_working(&mut self, doc_id: String, text: String) {
        let keywords = extract_keywords(&text);
        // Remove existing entry with same id
        self.working_items.retain(|item| item.doc_id != doc_id);
        self.working_items.push(WorkingItem {
            doc_id,
            text,
            keywords,
            age_seconds: 0,
        });
        // Cap working memory size
        while self.working_items.len() > 32 {
            self.working_items.remove(0);
        }
    }

    /// Remove from working memory.
    pub fn remove_working(&mut self, doc_id: &str) {
        self.working_items.retain(|item| item.doc_id != doc_id);
    }

    /// Age all working memory items by the given number of seconds.
    pub fn tick_working(&mut self, elapsed_secs: u64) {
        for item in &mut self.working_items {
            item.age_seconds += elapsed_secs;
        }
    }

    // ── Layer 2: Session Memory ──

    /// Index a session entry (e.g., from a recent agent turn).
    pub fn add_session(&mut self, doc_id: String, text: String) {
        let embedding = self.embedder.encode(&text);
        self.session_index.insert(doc_id, embedding, text);
    }

    /// Remove a session entry.
    pub fn remove_session(&mut self, doc_id: &str) {
        self.session_index.remove(doc_id);
    }

    /// Clear all session memory (e.g., at start of new day).
    pub fn clear_session(&mut self) {
        self.session_index.clear();
    }

    // ── Layer 3: Long-term Memory ──

    /// Index a vault note for long-term retrieval.
    pub fn add_longterm(&mut self, doc_id: String, text: String) {
        let embedding = self.embedder.encode(&text);
        self.longterm_index.insert(doc_id, embedding, text);
    }

    /// Remove from long-term index.
    pub fn remove_longterm(&mut self, doc_id: &str) {
        self.longterm_index.remove(doc_id);
    }

    // ── Search ──

    /// Progressive search across all 3 layers. Returns merged, deduplicated results.
    pub fn search(&self, query: &str) -> Vec<MemoryEntry> {
        let query_embedding = self.embedder.encode(query);
        let query_lower = query.to_lowercase();

        let mut results = Vec::new();

        // Layer 1: Working memory (keyword match, no embedding search)
        let working_results = self.search_working(&query_lower);
        results.extend(working_results);

        // Layer 2: Session memory (semantic search)
        let session_results = self
            .session_index
            .search(&query_embedding, self.config.session_limit);
        for r in session_results {
            results.push(MemoryEntry {
                doc_id: r.doc_id,
                text: r.text,
                score: r.score * MemoryLayer::Session.weight(),
                layer: MemoryLayer::Session,
                age_seconds: 0, // Session results don't track age yet
            });
        }

        // Layer 3: Long-term memory (semantic search)
        let longterm_results = self
            .longterm_index
            .search(&query_embedding, self.config.longterm_limit);
        for r in longterm_results {
            results.push(MemoryEntry {
                doc_id: r.doc_id,
                text: r.text,
                score: r.score * MemoryLayer::LongTerm.weight(),
                layer: MemoryLayer::LongTerm,
                age_seconds: 0,
            });
        }

        // Apply recency decay
        for entry in &mut results {
            if entry.age_seconds > 0 && self.config.recency_half_life_secs > 0 {
                let decay = 0.5_f64
                    .powf(entry.age_seconds as f64 / self.config.recency_half_life_secs as f64);
                entry.score *= decay;
            }
        }

        // Filter below threshold
        results.retain(|e| e.score >= self.config.min_score);

        // Deduplicate by doc_id (keep highest score)
        deduplicate_by_id(&mut results);

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to total limit
        results.truncate(self.config.total_limit);

        results
    }

    /// Count of entries across all layers.
    pub fn total_entries(&self) -> (usize, usize, usize) {
        (
            self.working_items.len(),
            self.session_index.len(),
            self.longterm_index.len(),
        )
    }

    // ── Internal ──

    fn search_working(&self, query_lower: &str) -> Vec<MemoryEntry> {
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        self.working_items
            .iter()
            .filter_map(|item| {
                let matches = item
                    .keywords
                    .iter()
                    .filter(|kw| {
                        query_words
                            .iter()
                            .any(|qw| kw.contains(qw) || qw.contains(kw.as_str()))
                    })
                    .count();

                if matches == 0 {
                    return None;
                }

                let keyword_score = matches as f64 / item.keywords.len().max(1) as f64;
                let score = keyword_score * MemoryLayer::Working.weight();

                Some(MemoryEntry {
                    doc_id: item.doc_id.clone(),
                    text: item.text.clone(),
                    score,
                    layer: MemoryLayer::Working,
                    age_seconds: item.age_seconds,
                })
            })
            .take(self.config.working_limit)
            .collect()
    }
}

/// Extract keywords from text (lowercase, deduplicated, no stop words).
fn extract_keywords(text: &str) -> Vec<String> {
    let stop_words = [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do",
        "does", "did", "will", "would", "can", "could", "to", "of", "in", "for", "on", "with",
        "at", "by", "from", "and", "but", "or", "not", "that", "this", "it", "its", "my", "your",
    ];

    let mut seen = std::collections::HashSet::new();
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2 && !stop_words.contains(w))
        .filter(|w| seen.insert(w.to_string()))
        .map(|w| w.to_string())
        .collect()
}

/// Deduplicate entries by doc_id, keeping the highest-scoring entry.
fn deduplicate_by_id(results: &mut Vec<MemoryEntry>) {
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut keep: Vec<MemoryEntry> = Vec::with_capacity(results.len());

    for entry in results.drain(..) {
        match seen.get(&entry.doc_id).copied() {
            Some(idx) => {
                // Keep the one with higher score
                if entry.score > keep[idx].score {
                    keep[idx] = entry;
                }
            }
            None => {
                seen.insert(entry.doc_id.clone(), keep.len());
                keep.push(entry);
            }
        }
    }

    *results = keep;
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ProgressiveConfig {
        ProgressiveConfig {
            working_limit: 3,
            session_limit: 5,
            longterm_limit: 5,
            total_limit: 10,
            min_score: 0.01,
            recency_half_life_secs: 3600,
        }
    }

    #[test]
    fn test_working_memory_add_remove() {
        let mut retriever = ProgressiveRetriever::new(test_config());
        retriever.add_working("w1".into(), "Rust programming systems".into());
        retriever.add_working("w2".into(), "Swift macOS development".into());

        let (w, s, l) = retriever.total_entries();
        assert_eq!(w, 2);
        assert_eq!(s, 0);
        assert_eq!(l, 0);

        retriever.remove_working("w1");
        let (w, _, _) = retriever.total_entries();
        assert_eq!(w, 1);
    }

    #[test]
    fn test_working_memory_search() {
        let mut retriever = ProgressiveRetriever::new(test_config());
        retriever.add_working("w1".into(), "Rust programming language".into());
        retriever.add_working("w2".into(), "Swift macOS native".into());

        let results = retriever.search("Rust programming");
        assert!(!results.is_empty());
        assert_eq!(results[0].layer, MemoryLayer::Working);
        assert!(results[0].doc_id == "w1");
    }

    #[test]
    fn test_session_memory_search() {
        let mut retriever = ProgressiveRetriever::new(test_config());
        retriever.add_session(
            "s1".into(),
            "agent completed research on quantum physics".into(),
        );
        retriever.add_session(
            "s2".into(),
            "agent deployed web application successfully".into(),
        );

        let results = retriever.search("quantum physics research");
        assert!(!results.is_empty());
        // Session results should have Session layer
        let session_results: Vec<_> = results
            .iter()
            .filter(|r| r.layer == MemoryLayer::Session)
            .collect();
        assert!(!session_results.is_empty());
    }

    #[test]
    fn test_longterm_memory_search() {
        let mut retriever = ProgressiveRetriever::new(test_config());
        retriever.add_longterm(
            "lt1".into(),
            "machine learning neural networks deep learning".into(),
        );
        retriever.add_longterm("lt2".into(), "Italian cooking pasta recipes".into());

        let results = retriever.search("neural networks deep learning");
        assert!(!results.is_empty());
        let lt_results: Vec<_> = results
            .iter()
            .filter(|r| r.layer == MemoryLayer::LongTerm)
            .collect();
        assert!(!lt_results.is_empty());
    }

    #[test]
    fn test_layer_priority_ordering() {
        let mut retriever = ProgressiveRetriever::new(test_config());

        // Add same-ish content to all 3 layers
        retriever.add_working("w1".into(), "Rust systems programming".into());
        retriever.add_session("s1".into(), "Rust systems programming notes".into());
        retriever.add_longterm("lt1".into(), "Rust systems programming guide".into());

        let results = retriever.search("Rust systems programming");
        // Working memory should be prioritized (highest weight)
        assert!(!results.is_empty());
        assert_eq!(results[0].layer, MemoryLayer::Working);
    }

    #[test]
    fn test_deduplication() {
        let mut retriever = ProgressiveRetriever::new(test_config());

        // Same doc_id in session and longterm
        retriever.add_session("shared-doc".into(), "quantum computing basics".into());
        retriever.add_longterm("shared-doc".into(), "quantum computing basics".into());

        let results = retriever.search("quantum computing");
        let shared_count = results.iter().filter(|r| r.doc_id == "shared-doc").count();
        assert_eq!(shared_count, 1, "Duplicate doc_ids should be merged");
    }

    #[test]
    fn test_recency_decay() {
        let mut retriever = ProgressiveRetriever::new(test_config());
        retriever.add_working("old".into(), "Rust programming guide".into());
        retriever.tick_working(7200); // 2 hours old

        retriever.add_working("new".into(), "Rust programming tutorial".into());
        // new has age_seconds = 0

        let results = retriever.search("Rust programming");
        // Both should be found, but new should score higher due to recency
        let old_entry = results.iter().find(|r| r.doc_id == "old");
        let new_entry = results.iter().find(|r| r.doc_id == "new");

        if let (Some(old), Some(new)) = (old_entry, new_entry) {
            assert!(new.score >= old.score, "Newer entries should score higher");
        }
    }

    #[test]
    fn test_total_limit() {
        let mut config = test_config();
        config.total_limit = 3;
        let mut retriever = ProgressiveRetriever::new(config);

        for i in 0..20 {
            retriever.add_longterm(
                format!("lt{i}"),
                format!("topic {i} about programming systems"),
            );
        }

        let results = retriever.search("programming systems");
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_empty_search() {
        let retriever = ProgressiveRetriever::new(test_config());
        let results = retriever.search("anything");
        assert!(results.is_empty());
    }

    #[test]
    fn test_clear_session() {
        let mut retriever = ProgressiveRetriever::new(test_config());
        retriever.add_session("s1".into(), "some session data".into());
        retriever.clear_session();
        let (_, s, _) = retriever.total_entries();
        assert_eq!(s, 0);
    }

    #[test]
    fn test_extract_keywords() {
        let kw = extract_keywords("The Rust programming language is great for systems");
        assert!(kw.contains(&"rust".to_string()));
        assert!(kw.contains(&"programming".to_string()));
        assert!(!kw.contains(&"the".to_string())); // stop word
        assert!(!kw.contains(&"is".to_string())); // stop word
    }

    #[test]
    fn test_memory_layer_as_str() {
        assert_eq!(MemoryLayer::Working.as_str(), "working");
        assert_eq!(MemoryLayer::Session.as_str(), "session");
        assert_eq!(MemoryLayer::LongTerm.as_str(), "long_term");
    }
}
