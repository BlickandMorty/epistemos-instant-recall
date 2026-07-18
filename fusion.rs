// Reciprocal Rank Fusion (RRF) + Cross-Encoder Reranking.
//
// Four-signal hybrid search pipeline:
//   1. tantivy FTS (BM25) → top-100
//   2. Vector search (segmented index) → top-100
//   3. Graph traversal (entity NER) → related chunks (handled externally)
//   4. RRF fusion → top-50
//   5. Cross-encoder reranking → top-10
//
// RRF formula: score(d) = Σ 1/(k + rank_r(d))
// where k=60 (standard constant from Cormack et al., SIGIR 2009).
//
// The cross-encoder is the final arbiter — it sees the full query-document
// pair and produces a relevance score. Uses ONNX Runtime with
// ms-marco-MiniLM-L-6-v2 (22MB) for sub-20ms reranking of 50 candidates.
//
// Target: full pipeline < 50ms on M2 Pro.

use std::collections::HashMap;

/// A retrieval result from a single signal (FTS, vector, graph).
#[derive(Debug, Clone)]
pub struct RetrievalHit {
    /// Document identifier.
    pub doc_id: String,
    /// Document text (or chunk).
    pub text: String,
    /// Score from this retrieval signal (higher = more relevant).
    pub score: f64,
    /// Which retrieval signal produced this hit.
    pub source: RetrievalSource,
}

/// Which retrieval signal produced a hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RetrievalSource {
    /// tantivy BM25 full-text search.
    FullText,
    /// Semantic vector search (segmented index).
    Vector,
    /// Knowledge graph traversal.
    Graph,
    /// External/custom source.
    Custom,
}

/// A fused result after RRF merging.
#[derive(Debug, Clone)]
pub struct FusedResult {
    pub doc_id: String,
    pub text: String,
    /// RRF score: Σ 1/(k + rank_r(d)) across all signals.
    pub rrf_score: f64,
    /// Which signals contributed to this result.
    pub sources: Vec<RetrievalSource>,
    /// Optional cross-encoder reranking score (set after reranking).
    pub rerank_score: Option<f64>,
}

/// Configuration for the fusion pipeline.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// RRF constant k. Default: 60 (Cormack et al. recommendation).
    pub rrf_k: f64,
    /// Weight multiplier for each source. Default: all 1.0.
    pub source_weights: HashMap<RetrievalSource, f64>,
    /// Maximum results from RRF fusion before reranking.
    pub fusion_top_k: usize,
    /// Maximum final results after reranking.
    pub final_top_k: usize,
}

impl Default for FusionConfig {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert(RetrievalSource::FullText, 0.5);
        weights.insert(RetrievalSource::Vector, 0.5);
        weights.insert(RetrievalSource::Graph, 0.3);
        weights.insert(RetrievalSource::Custom, 0.2);
        Self {
            rrf_k: 60.0,
            source_weights: weights,
            fusion_top_k: 50,
            final_top_k: 10,
        }
    }
}

/// Perform Reciprocal Rank Fusion across multiple retrieval signals.
///
/// Each signal provides a ranked list of hits. RRF merges them into a single
/// ranked list using: score(d) = Σ w_source × 1/(k + rank_r(d))
///
/// This is provably robust — it doesn't require score calibration between signals,
/// only rankings.
pub fn reciprocal_rank_fusion(
    signals: &[Vec<RetrievalHit>],
    config: &FusionConfig,
) -> Vec<FusedResult> {
    // Per-document accumulator: (rrf_score, text, sources)
    let mut doc_scores: HashMap<String, (f64, String, Vec<RetrievalSource>)> = HashMap::new();

    for signal_hits in signals {
        if signal_hits.is_empty() {
            continue;
        }

        let source = signal_hits[0].source;
        let weight = config.source_weights.get(&source).copied().unwrap_or(1.0);

        for (rank, hit) in signal_hits.iter().enumerate() {
            let rrf_contribution = weight / (config.rrf_k + (rank + 1) as f64);

            let entry = doc_scores
                .entry(hit.doc_id.clone())
                .or_insert_with(|| (0.0, hit.text.clone(), Vec::new()));
            entry.0 += rrf_contribution;
            if !entry.2.contains(&source) {
                entry.2.push(source);
            }
        }
    }

    // Sort by RRF score descending
    let mut results: Vec<FusedResult> = doc_scores
        .into_iter()
        .map(|(doc_id, (rrf_score, text, sources))| FusedResult {
            doc_id,
            text,
            rrf_score,
            sources,
            rerank_score: None,
        })
        .collect();

    results.sort_by(|a, b| {
        b.rrf_score
            .partial_cmp(&a.rrf_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(config.fusion_top_k);
    results
}

/// Cross-encoder reranking interface.
///
/// The cross-encoder takes (query, document) pairs and produces relevance scores.
/// This trait abstracts over the actual model (ONNX, CoreML, etc.) so tests
/// can use a mock implementation.
pub trait CrossEncoder: Send + Sync {
    /// Score a batch of (query, document) pairs.
    /// Returns scores in the same order as the input documents.
    /// Higher score = more relevant.
    fn score_batch(&self, query: &str, documents: &[&str]) -> Vec<f64>;
}

/// Rerank fused results using a cross-encoder.
///
/// Takes the top fusion_top_k results from RRF, scores each (query, doc) pair
/// with the cross-encoder, and returns the top final_top_k by rerank score.
pub fn cross_encoder_rerank(
    query: &str,
    mut fused: Vec<FusedResult>,
    encoder: &dyn CrossEncoder,
    final_top_k: usize,
) -> Vec<FusedResult> {
    if fused.is_empty() {
        return fused;
    }

    let documents: Vec<&str> = fused.iter().map(|r| r.text.as_str()).collect();
    let scores = encoder.score_batch(query, &documents);

    for (result, &score) in fused.iter_mut().zip(scores.iter()) {
        result.rerank_score = Some(score);
    }

    // Sort by rerank score descending
    fused.sort_by(|a, b| {
        b.rerank_score
            .unwrap_or(0.0)
            .partial_cmp(&a.rerank_score.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    fused.truncate(final_top_k);
    fused
}

/// BM25-based cross-encoder fallback when no neural model is available.
/// Uses term overlap scoring as a lightweight substitute.
/// This is NOT a real cross-encoder — it's a fallback for when ONNX isn't loaded.
pub struct BM25CrossEncoder {
    /// BM25 k1 parameter.
    k1: f64,
    /// BM25 b parameter.
    b: f64,
}

impl Default for BM25CrossEncoder {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

impl BM25CrossEncoder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl CrossEncoder for BM25CrossEncoder {
    fn score_batch(&self, query: &str, documents: &[&str]) -> Vec<f64> {
        let query_terms: Vec<&str> = query.split_whitespace().collect();
        let avg_dl: f64 = documents
            .iter()
            .map(|d| d.split_whitespace().count() as f64)
            .sum::<f64>()
            / documents.len().max(1) as f64;

        documents
            .iter()
            .map(|doc| {
                let doc_terms: Vec<&str> = doc.split_whitespace().collect();
                let dl = doc_terms.len() as f64;

                query_terms
                    .iter()
                    .map(|qt| {
                        let qt_lower = qt.to_lowercase();
                        let tf = doc_terms
                            .iter()
                            .filter(|dt| dt.to_lowercase() == qt_lower)
                            .count() as f64;
                        if tf == 0.0 {
                            return 0.0;
                        }
                        // Simplified BM25 (no IDF since we don't have corpus stats)
                        (tf * (self.k1 + 1.0))
                            / (tf + self.k1 * (1.0 - self.b + self.b * dl / avg_dl))
                    })
                    .sum()
            })
            .collect()
    }
}

/// The full hybrid search pipeline.
///
/// Orchestrates: FTS → Vector → Graph → RRF → Rerank → Final results.
/// This struct holds the configuration; actual signal retrieval is done by callers.
pub struct HybridSearchPipeline {
    pub config: FusionConfig,
    pub cross_encoder: Box<dyn CrossEncoder>,
}

impl HybridSearchPipeline {
    /// Create a pipeline with a BM25 fallback cross-encoder.
    pub fn with_bm25_fallback(config: FusionConfig) -> Self {
        Self {
            config,
            cross_encoder: Box::new(BM25CrossEncoder::new()),
        }
    }

    /// Create a pipeline with a custom cross-encoder.
    pub fn with_encoder(config: FusionConfig, encoder: Box<dyn CrossEncoder>) -> Self {
        Self {
            config,
            cross_encoder: encoder,
        }
    }

    /// Execute the full pipeline: fuse signals → rerank → return top results.
    pub fn execute(&self, query: &str, signals: &[Vec<RetrievalHit>]) -> Vec<FusedResult> {
        let fused = reciprocal_rank_fusion(signals, &self.config);
        cross_encoder_rerank(
            query,
            fused,
            self.cross_encoder.as_ref(),
            self.config.final_top_k,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fts_hits() -> Vec<RetrievalHit> {
        vec![
            RetrievalHit {
                doc_id: "doc-a".into(),
                text: "Rust systems programming".into(),
                score: 5.0,
                source: RetrievalSource::FullText,
            },
            RetrievalHit {
                doc_id: "doc-b".into(),
                text: "Swift macOS development".into(),
                score: 4.0,
                source: RetrievalSource::FullText,
            },
            RetrievalHit {
                doc_id: "doc-c".into(),
                text: "Python data science".into(),
                score: 3.0,
                source: RetrievalSource::FullText,
            },
        ]
    }

    fn make_vector_hits() -> Vec<RetrievalHit> {
        vec![
            RetrievalHit {
                doc_id: "doc-b".into(),
                text: "Swift macOS development".into(),
                score: 0.9,
                source: RetrievalSource::Vector,
            },
            RetrievalHit {
                doc_id: "doc-a".into(),
                text: "Rust systems programming".into(),
                score: 0.8,
                source: RetrievalSource::Vector,
            },
            RetrievalHit {
                doc_id: "doc-d".into(),
                text: "Machine learning neural nets".into(),
                score: 0.7,
                source: RetrievalSource::Vector,
            },
        ]
    }

    #[test]
    fn rrf_merges_signals() {
        let config = FusionConfig::default();
        let fts = make_fts_hits();
        let vec = make_vector_hits();

        let fused = reciprocal_rank_fusion(&[fts, vec], &config);

        // doc-a and doc-b appear in both signals, should rank high
        assert!(!fused.is_empty());
        let top_ids: Vec<&str> = fused.iter().take(2).map(|r| r.doc_id.as_str()).collect();
        assert!(
            top_ids.contains(&"doc-a") || top_ids.contains(&"doc-b"),
            "Documents in both signals should rank highly"
        );
    }

    #[test]
    fn rrf_handles_single_signal() {
        let config = FusionConfig::default();
        let fts = make_fts_hits();
        let fused = reciprocal_rank_fusion(&[fts], &config);
        assert_eq!(fused.len(), 3);
    }

    #[test]
    fn rrf_handles_empty_signals() {
        let config = FusionConfig::default();
        let fused = reciprocal_rank_fusion(&[], &config);
        assert!(fused.is_empty());
    }

    #[test]
    fn rrf_multi_source_tracking() {
        let config = FusionConfig::default();
        let fts = make_fts_hits();
        let vec = make_vector_hits();
        let fused = reciprocal_rank_fusion(&[fts, vec], &config);

        // doc-a appears in both
        let doc_a = fused.iter().find(|r| r.doc_id == "doc-a").unwrap();
        assert!(doc_a.sources.contains(&RetrievalSource::FullText));
        assert!(doc_a.sources.contains(&RetrievalSource::Vector));
        assert_eq!(doc_a.sources.len(), 2);

        // doc-d only in vector
        let doc_d = fused.iter().find(|r| r.doc_id == "doc-d").unwrap();
        assert_eq!(doc_d.sources.len(), 1);
        assert_eq!(doc_d.sources[0], RetrievalSource::Vector);
    }

    #[test]
    fn bm25_cross_encoder_scores() {
        let encoder = BM25CrossEncoder::new();
        let scores = encoder.score_batch(
            "Rust programming",
            &[
                "Rust systems programming language",
                "Python data science",
                "Rust and Go comparison",
            ],
        );
        assert_eq!(scores.len(), 3);
        // First doc should score highest (contains both "Rust" and "programming")
        assert!(
            scores[0] > scores[1],
            "Doc with query terms should score higher"
        );
    }

    #[test]
    fn cross_encoder_rerank_reorders() {
        let fused = vec![
            FusedResult {
                doc_id: "low".into(),
                text: "cooking recipes".into(),
                rrf_score: 0.5,
                sources: vec![RetrievalSource::FullText],
                rerank_score: None,
            },
            FusedResult {
                doc_id: "high".into(),
                text: "Rust programming systems".into(),
                rrf_score: 0.3,
                sources: vec![RetrievalSource::Vector],
                rerank_score: None,
            },
        ];

        let encoder = BM25CrossEncoder::new();
        let reranked = cross_encoder_rerank("Rust programming", fused, &encoder, 2);

        assert_eq!(reranked.len(), 2);
        // "high" doc should now be first due to cross-encoder scoring
        assert_eq!(reranked[0].doc_id, "high");
        assert!(reranked[0].rerank_score.is_some());
    }

    #[test]
    fn full_pipeline_end_to_end() {
        let config = FusionConfig {
            final_top_k: 2,
            ..Default::default()
        };
        let pipeline = HybridSearchPipeline::with_bm25_fallback(config);

        let fts = make_fts_hits();
        let vec = make_vector_hits();

        let results = pipeline.execute("Rust systems programming", &[fts, vec]);
        assert!(results.len() <= 2);
        assert!(results[0].rerank_score.is_some());
    }

    #[test]
    fn rrf_respects_source_weights() {
        let mut config = FusionConfig::default();
        config.source_weights.insert(RetrievalSource::FullText, 2.0); // Double FTS weight
        config.source_weights.insert(RetrievalSource::Vector, 0.5);

        // doc-c is rank 3 in FTS only, doc-d is rank 3 in vector only
        let fts = make_fts_hits();
        let vec = make_vector_hits();
        let fused = reciprocal_rank_fusion(&[fts, vec], &config);

        let doc_c = fused.iter().find(|r| r.doc_id == "doc-c").unwrap();
        let doc_d = fused.iter().find(|r| r.doc_id == "doc-d").unwrap();

        // doc-c should score higher because FTS is weighted 2x
        assert!(
            doc_c.rrf_score > doc_d.rrf_score,
            "FTS-only doc with 2x weight should beat vector-only doc with 0.5x weight"
        );
    }
}
