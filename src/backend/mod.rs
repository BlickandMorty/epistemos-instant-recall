//! Shadow backend trait and edition-selected production backend.
//!
//! Free V1 compiles only the deterministic Tantivy/BM25 note backend.
//! A semantic backend remains separately selectable for a future edition,
//! but is mutually exclusive with the Free feature.

#[cfg(all(feature = "free-lexical", feature = "semantic"))]
compile_error!("free-lexical and semantic are mutually exclusive epistemos-shadow features");

#[cfg(not(any(feature = "free-lexical", feature = "semantic")))]
compile_error!("select exactly one epistemos-shadow backend feature");

#[cfg(feature = "semantic")]
pub mod embedder;
pub mod free_semantic;
pub mod lexical_index;
#[cfg(feature = "semantic")]
pub mod rrf;
#[cfg(feature = "semantic")]
pub mod vector_index;

#[cfg(feature = "free-lexical")]
mod free_backend;
#[cfg(feature = "free-lexical")]
pub use free_backend::RealBackend;

#[cfg(feature = "semantic")]
use std::path::{Path, PathBuf};
#[cfg(feature = "semantic")]
use std::sync::{Mutex, RwLock};
#[cfg(feature = "semantic")]
use std::time::Instant;

#[cfg(feature = "semantic")]
use rustc_hash::FxHashMap;

use crate::error::ShadowError;
use crate::{ShadowDocument, ShadowHit, ShadowStats};

#[cfg(feature = "semantic")]
use embedder::Embedder;
#[cfg(feature = "semantic")]
use lexical_index::LexicalIndex;
#[cfg(feature = "semantic")]
use vector_index::VectorIndex;

/// Pluggable backend for the Shadow engine. The FFI surface
/// (`shadow_insert_json` / `shadow_remove_json` / `shadow_search_json`
/// / `shadow_flush` / `shadow_stats_json` in lib.rs) dispatches every
/// call through this trait. `RealBackend` is the production
/// implementor; `state::ShadowState` is the in-memory fallback used
/// before `shadow_open_at` succeeds and by deterministic tests.
///
/// Trait is `Send + Sync` because `lib.rs` reaches the singleton
/// across the FFI boundary on whatever thread the Swift caller hops
/// onto. Implementors that need interior mutability must use
/// `parking_lot::RwLock` or `tokio::sync` to honor that contract.
pub trait ShadowBackend: Send + Sync {
    fn insert_document(&self, doc: ShadowDocument) -> Result<(), ShadowError>;
    fn remove_document(&self, doc_id: &str) -> Result<(), ShadowError>;
    #[cfg(feature = "free-lexical")]
    fn search_notes(&self, query: &str, limit: usize) -> Result<Vec<ShadowHit>, ShadowError>;
    #[cfg(feature = "semantic")]
    fn search(
        &self,
        query: &str,
        domain: &str,
        limit: usize,
    ) -> Result<Vec<ShadowHit>, ShadowError>;
    fn flush(&self) -> Result<(), ShadowError>;
    fn stats(&self) -> Result<ShadowStats, ShadowError>;
}

// MARK: - RealBackend (W8.4.e — the production implementor)
//
// Owns one shared Embedder (Model2Vec, lazy-init via Embedder::global()),
// one LexicalIndex (tantivy multi-domain, filtered at query time), and
// per-domain VectorIndex (one for "note", one for "chat" — keeps
// removals O(log n) without needing usearch metadata filtering).
//
// Thread-safe per the ShadowBackend contract: every mutating method
// hops through `Mutex` / `RwLock` guards before touching the index
// state. Reads are concurrent (RwLock::read).

#[cfg(feature = "semantic")]
pub struct RealBackend {
    embedder: &'static Embedder,
    lexical: LexicalIndex,
    vectors: RwLock<FxHashMap<String, VectorIndex>>,
    /// Snippet hydration map — doc_id → ShadowDocument so search can
    /// emit `snippet` + `title` without re-querying tantivy.
    docs: RwLock<FxHashMap<String, ShadowDocument>>,
    last_flush: Mutex<Instant>,
    /// Optional persistence root. `None` for the in-memory variant
    /// (test fixtures + the `new()` constructor). When set, `flush()`
    /// writes the tantivy + usearch sidecars under this path so the
    /// next `open(path)` call restores the index in place.
    persistence_root: Option<PathBuf>,
}

#[cfg(feature = "semantic")]
impl RealBackend {
    /// Construct a fresh in-memory RealBackend. Triggers HuggingFace
    /// download on first call (~30 MB cached at
    /// ~/.cache/huggingface/hub/). For production use,
    /// `open_at(path)` is the canonical constructor — that variant
    /// persists across app restarts.
    pub fn new() -> Result<Self, ShadowError> {
        let embedder = Embedder::global()?;
        let lexical = LexicalIndex::new()?;
        Ok(Self {
            embedder,
            lexical,
            vectors: RwLock::new(FxHashMap::default()),
            docs: RwLock::new(FxHashMap::default()),
            last_flush: Mutex::new(Instant::now()),
            persistence_root: None,
        })
    }

    /// Open (or create) a RealBackend rooted at `path`. The vault
    /// layout under `path` is:
    ///
    ///   path/
    ///     tantivy/                    MmapDirectory for BM25
    ///     vectors/<domain>.usearch    HNSW sidecar per domain
    ///     vectors/<domain>.mapping.json   doc_id ↔ row_key map
    ///     docs.json                   snippet hydration cache
    ///
    /// On startup, all four are loaded back into memory so the next
    /// `search()` call hits a hot index. First-launch (path didn't
    /// exist) returns an empty backend ready for inserts.
    pub fn open_at(path: &Path) -> Result<Self, ShadowError> {
        let embedder = Embedder::global()?;
        std::fs::create_dir_all(path).map_err(|e| ShadowError::Io {
            detail: format!("create_dir_all({path:?}) failed: {e}"),
        })?;
        let tantivy_path = path.join("tantivy");
        let lexical = LexicalIndex::open_at(&tantivy_path)?;

        // Restore docs side map (snippet hydration). First-launch path
        // → no docs.json → empty map.
        let docs_path = path.join("docs.json");
        let docs: FxHashMap<String, ShadowDocument> = if docs_path.exists() {
            let bytes = std::fs::read(&docs_path).map_err(|e| ShadowError::Io {
                detail: format!("read({docs_path:?}) failed: {e}"),
            })?;
            serde_json::from_slice(&bytes).map_err(|e| ShadowError::Backend {
                detail: format!("docs.json decode failed: {e}"),
            })?
        } else {
            FxHashMap::default()
        };

        // Restore per-domain vector indices. Walk the lexical's stored
        // doc_ids to discover which domains exist, then load each one's
        // sidecar pair (vectors + mapping).
        let mut vectors: FxHashMap<String, VectorIndex> = FxHashMap::default();
        let observed = lexical.iter_doc_ids()?;
        let mut domains: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (_doc_id, domain) in &observed {
            domains.insert(domain.clone());
        }
        let vectors_dir = path.join("vectors");
        for domain in domains {
            let usearch_path = vectors_dir.join(format!("{domain}.usearch"));
            let mapping_path = vectors_dir.join(format!("{domain}.mapping.json"));
            if !usearch_path.exists() || !mapping_path.exists() {
                continue;
            }
            let index = VectorIndex::new(embedder::EMBED_DIM)?;
            index.load_from(&usearch_path)?;
            index.load_mapping_from(&mapping_path)?;
            vectors.insert(domain, index);
        }

        Ok(Self {
            embedder,
            lexical,
            vectors: RwLock::new(vectors),
            docs: RwLock::new(docs),
            last_flush: Mutex::new(Instant::now()),
            persistence_root: Some(path.to_path_buf()),
        })
    }

    fn ensure_vector_index(&self, domain: &str) -> Result<(), ShadowError> {
        let read_guard = self.vectors.read().expect("vectors lock poisoned");
        if read_guard.contains_key(domain) {
            return Ok(());
        }
        drop(read_guard);
        let mut write_guard = self.vectors.write().expect("vectors lock poisoned");
        if write_guard.contains_key(domain) {
            return Ok(()); // raced; someone else created it
        }
        let index = VectorIndex::new(embedder::EMBED_DIM)?;
        write_guard.insert(domain.to_string(), index);
        Ok(())
    }
}

#[cfg(feature = "semantic")]
impl ShadowBackend for RealBackend {
    fn insert_document(&self, doc: ShadowDocument) -> Result<(), ShadowError> {
        if doc.doc_id.is_empty() {
            return Err(ShadowError::InvalidInput {
                detail: "doc_id was empty".into(),
            });
        }
        if doc.domain != "note" && doc.domain != "chat" {
            return Err(ShadowError::InvalidInput {
                detail: format!(
                    "unknown domain '{}' (expected 'note' or 'chat')",
                    doc.domain
                ),
            });
        }

        // Embed the body (title concatenation is the V1 retrieval
        // strategy — title carries strong semantic signal).
        let combined = format!("{}\n{}", doc.title, doc.body);
        let embedding = self.embedder.encode_one(&combined);
        if embedding.len() != embedder::EMBED_DIM {
            return Err(ShadowError::Backend {
                detail: format!(
                    "embedder returned {} dims; expected {}",
                    embedding.len(),
                    embedder::EMBED_DIM
                ),
            });
        }

        self.ensure_vector_index(&doc.domain)?;
        {
            let vectors = self.vectors.read().expect("vectors lock poisoned");
            let index = vectors
                .get(&doc.domain)
                .ok_or_else(|| ShadowError::Backend {
                    detail: format!("vector index for '{}' missing post-ensure", doc.domain),
                })?;
            index.add(&doc.doc_id, &embedding)?;
        }

        self.lexical.insert(&doc)?;

        let mut docs = self.docs.write().expect("docs lock poisoned");
        docs.insert(doc.doc_id.clone(), doc);
        Ok(())
    }

    fn remove_document(&self, doc_id: &str) -> Result<(), ShadowError> {
        let mut docs = self.docs.write().expect("docs lock poisoned");
        let removed = docs.remove(doc_id);
        let Some(removed_doc) = removed else {
            return Err(ShadowError::NotFound {
                doc_id: doc_id.to_string(),
            });
        };

        if let Some(index) = self
            .vectors
            .read()
            .expect("vectors lock poisoned")
            .get(&removed_doc.domain)
        {
            index.remove(doc_id)?;
        }
        self.lexical.remove(doc_id)?;
        Ok(())
    }

    fn search(
        &self,
        query: &str,
        domain: &str,
        limit: usize,
    ) -> Result<Vec<ShadowHit>, ShadowError> {
        if domain != "note" && domain != "chat" {
            return Err(ShadowError::InvalidInput {
                detail: format!("unknown search domain '{domain}' (expected 'note' or 'chat')"),
            });
        }
        if query.trim().is_empty() || limit == 0 {
            return Ok(Vec::new());
        }

        // Encode + dense search through the per-domain VectorIndex.
        let dense_hits: Vec<(String, f32)> = {
            let vectors = self.vectors.read().expect("vectors lock poisoned");
            match vectors.get(domain) {
                Some(index) => {
                    let q_vec = self.embedder.encode_one(query);
                    if q_vec.is_empty() {
                        Vec::new()
                    } else {
                        index.search(&q_vec, limit * 2)
                    }
                }
                None => Vec::new(),
            }
        };

        let lexical_hits: Vec<(String, f32)> = self
            .lexical
            .search(query, domain, limit * 2)?
            .into_iter()
            .map(|h| (h.doc_id, h.score))
            .collect();

        // RRF fuse the two channels.
        let fused = rrf::rrf_fuse(&dense_hits, &lexical_hits, rrf::RRF_K_DEFAULT, limit);

        // Hydrate with snippet + title from the docs side map.
        let docs = self.docs.read().expect("docs lock poisoned");
        let dense_set: std::collections::HashSet<&str> =
            dense_hits.iter().map(|(id, _)| id.as_str()).collect();
        let lex_set: std::collections::HashSet<&str> =
            lexical_hits.iter().map(|(id, _)| id.as_str()).collect();

        let hits: Vec<ShadowHit> = fused
            .into_iter()
            .filter_map(|(doc_id, rrf_score)| {
                let doc = docs.get(&doc_id)?;
                let source = match (
                    dense_set.contains(doc_id.as_str()),
                    lex_set.contains(doc_id.as_str()),
                ) {
                    (true, true) => "rrf",
                    (true, false) => "dense",
                    (false, true) => "lexical",
                    (false, false) => "rrf", // unreachable in practice
                };
                Some(ShadowHit {
                    doc_id: doc.doc_id.clone(),
                    title: doc.title.clone(),
                    snippet: build_snippet(&doc.body, query),
                    score: rrf_score,
                    source: source.to_string(),
                    // Mirror the indexed sidecar metadata back to the
                    // search caller so the Swift host can apply the
                    // same vault-filter contract used by the graph.
                    origin_vault_key: doc.origin_vault_key.clone(),
                })
            })
            .collect();

        Ok(hits)
    }

    fn flush(&self) -> Result<(), ShadowError> {
        if let Some(root) = self.persistence_root.as_ref() {
            // Persist docs side map for snippet hydration.
            let docs = self.docs.read().expect("docs lock poisoned");
            let docs_path = root.join("docs.json");
            let bytes = serde_json::to_vec(&*docs).map_err(|e| ShadowError::Backend {
                detail: format!("docs.json encode failed: {e}"),
            })?;
            std::fs::write(&docs_path, bytes).map_err(|e| ShadowError::Io {
                detail: format!("write({docs_path:?}) failed: {e}"),
            })?;

            // Persist each per-domain vector index pair.
            let vectors_dir = root.join("vectors");
            std::fs::create_dir_all(&vectors_dir).map_err(|e| ShadowError::Io {
                detail: format!("create_dir_all({vectors_dir:?}) failed: {e}"),
            })?;
            let vectors = self.vectors.read().expect("vectors lock poisoned");
            for (domain, index) in vectors.iter() {
                let usearch_path = vectors_dir.join(format!("{domain}.usearch"));
                let mapping_path = vectors_dir.join(format!("{domain}.mapping.json"));
                index.save_to(&usearch_path)?;
                index.save_mapping_to(&mapping_path)?;
            }
            // tantivy commits per-insert (manual reload), so the
            // MmapDirectory is already on disk; nothing to flush there.
        }
        let mut last = self.last_flush.lock().expect("flush lock poisoned");
        *last = Instant::now();
        Ok(())
    }

    fn stats(&self) -> Result<ShadowStats, ShadowError> {
        let docs = self.docs.read().expect("docs lock poisoned");
        let mut note_count: u64 = 0;
        let mut chat_count: u64 = 0;
        let mut bytes: u64 = 0;
        for doc in docs.values() {
            bytes += (doc.title.len() + doc.body.len()) as u64;
            match doc.domain.as_str() {
                "note" => note_count += 1,
                "chat" => chat_count += 1,
                _ => {}
            }
        }
        let last_flush = self.last_flush.lock().expect("flush lock poisoned");
        Ok(ShadowStats {
            note_count,
            chat_count,
            index_size_bytes: bytes,
            last_flush_ms_ago: last_flush.elapsed().as_millis() as u64,
        })
    }
}

/// Pre-truncated snippet centred on the query match (or doc head when
/// no match). Mirrors the in-memory fallback algorithm so the Swift
/// inspector's snippet display is unchanged.
#[cfg(feature = "semantic")]
fn build_snippet(body: &str, query: &str) -> String {
    const MAX: usize = 160;
    if body.len() <= MAX {
        return body.to_string();
    }
    let body_lower = body.to_lowercase();
    let query_lower = query.to_lowercase();
    let center = body_lower.find(&query_lower).unwrap_or(0);
    let half = MAX / 2;
    let start = center.saturating_sub(half);
    let end = (start + MAX).min(body.len());
    let safe_start = (0..=start)
        .rev()
        .find(|i| body.is_char_boundary(*i))
        .unwrap_or(0);
    let safe_end = (end..=body.len())
        .find(|i| body.is_char_boundary(*i))
        .unwrap_or(body.len());
    body[safe_start..safe_end].to_string()
}

#[cfg(all(test, feature = "semantic"))]
mod tests {
    use super::*;
    use crate::state::ShadowState;

    #[test]
    fn shadow_state_satisfies_shadow_backend_trait() {
        // Regression guard for the W8.4.a contract.
        fn assert_trait<T: ShadowBackend>(_t: &T) {}
        let state = ShadowState::default();
        assert_trait(&state);
    }

    #[test]
    fn real_backend_satisfies_shadow_backend_trait() {
        // RealBackend MUST satisfy the trait without ever constructing
        // an instance — type-check only, doesn't call new() (which
        // would trigger Model2Vec download).
        fn assert_trait<T: ShadowBackend>() {}
        assert_trait::<RealBackend>();
    }

    #[test]
    #[ignore = "requires Model2Vec download (~30MB); run with --include-ignored"]
    fn real_backend_insert_then_search_returns_hit() {
        let backend = RealBackend::new().expect("RealBackend::new must succeed");
        let doc = ShadowDocument {
            doc_id: "doc-1".to_string(),
            domain: "note".to_string(),
            title: "Quarterly Report".to_string(),
            body: "Revenue grew by 12 percent across all regions.".to_string(),
            origin_vault_key: None,
        };
        backend.insert_document(doc).unwrap();

        let hits = backend.search("revenue", "note", 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-1");
        assert!(hits[0].score > 0.0);
        // Fused source — the doc shows up in BOTH dense (semantic
        // similarity) and lexical (exact "revenue" match) channels.
        assert_eq!(hits[0].source, "rrf");
    }

    #[test]
    #[ignore = "requires Model2Vec download (~30MB); run with --include-ignored"]
    fn real_backend_persistence_round_trips_search_results() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().to_path_buf();

        // Build at the path, insert + flush
        {
            let backend =
                RealBackend::open_at(&path).expect("first-open at empty path must succeed");
            backend
                .insert_document(ShadowDocument {
                    doc_id: "doc-1".to_string(),
                    domain: "note".to_string(),
                    title: "Persistent quarterly".to_string(),
                    body: "Revenue grew 12 percent across all regions.".to_string(),
                    origin_vault_key: None,
                })
                .unwrap();
            backend.flush().unwrap();
        }

        // Re-open at the same path and verify search works
        {
            let restored =
                RealBackend::open_at(&path).expect("second-open of populated path must succeed");
            let hits = restored.search("revenue", "note", 5).unwrap();
            assert_eq!(hits.len(), 1, "restored backend MUST find the doc");
            assert_eq!(hits[0].doc_id, "doc-1");
            // Stats survive too
            let stats = restored.stats().unwrap();
            assert_eq!(stats.note_count, 1);
        }
    }

    #[test]
    #[ignore = "requires Model2Vec download (~30MB); run with --include-ignored"]
    fn real_backend_search_filters_by_domain() {
        let backend = RealBackend::new().unwrap();
        backend
            .insert_document(ShadowDocument {
                doc_id: "n".into(),
                domain: "note".into(),
                title: "x".into(),
                body: "report".into(),
                origin_vault_key: None,
            })
            .unwrap();
        backend
            .insert_document(ShadowDocument {
                doc_id: "c".into(),
                domain: "chat".into(),
                title: "y".into(),
                body: "report".into(),
                origin_vault_key: None,
            })
            .unwrap();
        let note_hits = backend.search("report", "note", 5).unwrap();
        let chat_hits = backend.search("report", "chat", 5).unwrap();
        assert_eq!(note_hits.len(), 1);
        assert_eq!(note_hits[0].doc_id, "n");
        assert_eq!(chat_hits.len(), 1);
        assert_eq!(chat_hits[0].doc_id, "c");
    }
}
