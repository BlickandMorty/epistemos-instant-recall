//! W8.4.c — usearch HNSW vector index wrapper.
//!
//! Mirrors the production HNSW config at
//! `graph-engine/src/retrieval_index.rs:401-411`:
//!   metric:           Cos
//!   quantization:     F16   (4× memory saving vs F32, negligible
//!                            recall loss at 256-d for the Halo budget)
//!   connectivity:     16    (M parameter — Cormack/Malkov default)
//!   expansion_add:    128   (ef_construction)
//!   expansion_search: 64    (ef_search)
//!
//! Distance → similarity: `1.0 - distance` clamped to [-1, 1] mirrors
//! `retrieval_index.rs:517-519`. usearch returns L2/Cos distance per
//! the metric chosen; we convert so callers see "higher = more similar".
//!
//! ## doc_id ↔ row_key mapping
//!
//! usearch's `Index::add(key: u64, vec)` requires a numeric key. The
//! Halo doc ids are strings (ULIDs). We maintain two sibling
//! `FxHashMap`s:
//!   doc_to_key:  String → u64
//!   key_to_doc:  u64    → String
//! Plus a recycled `free_keys: Vec<u64>` so removal doesn't leak the
//! key space; new inserts pull from the free list before incrementing
//! `next_key`.

use std::path::Path;
use std::sync::RwLock;

use rustc_hash::FxHashMap;
use usearch::{
    Index,
    ffi::{IndexOptions, MetricKind, ScalarKind},
};

use crate::error::ShadowError;

/// HNSW connectivity parameter (Malkov's `M`). Mirrors graph-engine's
/// production config — not retuned for the Halo workload because the
/// W8.4 day-1 spike (commit 7c867f55) confirmed Model2Vec encoding
/// alone is 8× under budget; HNSW search at this M is comfortable.
pub const HNSW_CONNECTIVITY: usize = 16;
pub const HNSW_EXPANSION_ADD: usize = 128;
pub const HNSW_EXPANSION_SEARCH: usize = 64;

/// Initial reserve capacity. usearch's `reserve(n)` is amortised; we
/// pick 1024 so a typical first-launch vault scan doesn't trigger a
/// reserve until the second-thousand doc.
const INITIAL_RESERVE: usize = 1024;
/// When we exhaust the reserved capacity, grow by this factor.
const RESERVE_GROWTH_FACTOR: usize = 2;

pub struct VectorIndex {
    index: Index,
    /// Per-instance dimension; queries with a different shape return
    /// no hits rather than panicking (mirrors retrieval_index.rs).
    dimension: usize,
    /// Mutable mapping state behind a single RwLock so add/remove +
    /// search can interleave (search holds a read lock; add/remove
    /// upgrades to a write lock).
    state: RwLock<MappingState>,
}

#[derive(Default)]
struct MappingState {
    doc_to_key: FxHashMap<String, u64>,
    key_to_doc: FxHashMap<u64, String>,
    free_keys: Vec<u64>,
    next_key: u64,
    reserved_capacity: usize,
}

/// Disk-serializable form of MappingState — FxHashMap doesn't impl
/// Serialize directly so we flatten to a Vec of pairs for the JSON
/// sidecar (W8.4.f).
#[derive(serde::Serialize, serde::Deserialize)]
struct MappingSnapshot {
    doc_to_key: Vec<(String, u64)>,
    free_keys: Vec<u64>,
    next_key: u64,
    reserved_capacity: usize,
}

impl VectorIndex {
    /// Build a new HNSW index for the given dimension. Mirrors
    /// `graph-engine/src/retrieval_index.rs::new_index` exactly.
    pub fn new(dimension: usize) -> Result<Self, ShadowError> {
        if dimension == 0 {
            return Err(ShadowError::Backend {
                detail: "VectorIndex dimension must be > 0".into(),
            });
        }
        let options = IndexOptions {
            dimensions: dimension,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F16,
            connectivity: HNSW_CONNECTIVITY,
            expansion_add: HNSW_EXPANSION_ADD,
            expansion_search: HNSW_EXPANSION_SEARCH,
            multi: false,
        };
        let index = Index::new(&options).map_err(|e| ShadowError::Backend {
            detail: format!("usearch::Index::new failed: {e}"),
        })?;
        index
            .reserve(INITIAL_RESERVE)
            .map_err(|e| ShadowError::Backend {
                detail: format!("usearch reserve failed: {e}"),
            })?;
        Ok(Self {
            index,
            dimension,
            state: RwLock::new(MappingState {
                reserved_capacity: INITIAL_RESERVE,
                ..MappingState::default()
            }),
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn len(&self) -> usize {
        self.state
            .read()
            .expect("vector index lock poisoned")
            .doc_to_key
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert (or replace) a vector for `doc_id`. With `multi: false`
    /// usearch rejects duplicate keys, so a "replace" is implemented
    /// as remove-then-add. The mapping state preserves the doc_id's
    /// row key across the cycle so we don't leak the keyspace.
    pub fn add(&self, doc_id: &str, embedding: &[f32]) -> Result<(), ShadowError> {
        if embedding.len() != self.dimension {
            return Err(ShadowError::Backend {
                detail: format!(
                    "vector dim {} does not match index dim {}",
                    embedding.len(),
                    self.dimension
                ),
            });
        }
        let (key, is_replace) = self.allocate_or_reuse_key(doc_id);
        if is_replace {
            // Best-effort drop the old vector so usearch's
            // unique-key invariant accepts the next add.
            let _ = self.index.remove(key);
        }
        self.ensure_reserve(key)?;
        self.index
            .add::<f32>(key, embedding)
            .map_err(|e| ShadowError::Backend {
                detail: format!("usearch add(key={key}) failed: {e}"),
            })?;
        Ok(())
    }

    /// Remove the vector for `doc_id` if present. Idempotent — returns
    /// `Ok(())` whether the doc was found or not so the caller can use
    /// remove + insert as a "replace" without first checking presence.
    pub fn remove(&self, doc_id: &str) -> Result<(), ShadowError> {
        let mut guard = self.state.write().expect("vector index lock poisoned");
        let Some(key) = guard.doc_to_key.remove(doc_id) else {
            return Ok(());
        };
        guard.key_to_doc.remove(&key);
        guard.free_keys.push(key);
        // usearch supports per-key remove. Errors here mean the index
        // already lost the key, which is benign in this idempotent path.
        let _ = self.index.remove(key);
        Ok(())
    }

    /// Search the top `limit` nearest doc_ids for the query vector.
    /// Returns (doc_id, similarity) tuples where similarity is in
    /// `[-1, 1]` (1.0 = identical for cosine). Mismatched dimensions
    /// return an empty vec instead of erroring (mirrors
    /// retrieval_index.rs's permissive contract).
    pub fn search(&self, query: &[f32], limit: usize) -> Vec<(String, f32)> {
        if query.len() != self.dimension || limit == 0 {
            return Vec::new();
        }
        let Ok(matches) = self.index.search::<f32>(query, limit) else {
            return Vec::new();
        };
        let guard = self.state.read().expect("vector index lock poisoned");
        let mut hits: Vec<(String, f32)> = Vec::with_capacity(matches.keys.len());
        for (key, distance) in matches.keys.iter().zip(matches.distances.iter()) {
            let Some(doc_id) = guard.key_to_doc.get(key) else {
                continue;
            };
            // Cosine distance → similarity: `1 - distance`, clamped.
            let similarity = (1.0 - distance).clamp(-1.0, 1.0);
            hits.push((doc_id.clone(), similarity));
        }
        hits
    }

    // MARK: - internals

    /// Get-or-mint the row key for a doc_id. Reuses a recycled key
    /// from `free_keys` before incrementing `next_key` so the key
    /// space stays compact. Returns `(key, is_replace)` where
    /// `is_replace` is true iff the doc_id already had a row.
    fn allocate_or_reuse_key(&self, doc_id: &str) -> (u64, bool) {
        let mut guard = self.state.write().expect("vector index lock poisoned");
        if let Some(&existing) = guard.doc_to_key.get(doc_id) {
            return (existing, true);
        }
        let key = if let Some(recycled) = guard.free_keys.pop() {
            recycled
        } else {
            let k = guard.next_key;
            guard.next_key += 1;
            k
        };
        guard.doc_to_key.insert(doc_id.to_string(), key);
        guard.key_to_doc.insert(key, doc_id.to_string());
        (key, false)
    }

    // MARK: - Disk persistence (W8.4.f)

    /// Persist the HNSW vectors to a `.usearch` sidecar at `path`.
    /// usearch's native serializer; round-trips losslessly. The
    /// caller MUST also persist the doc_id ↔ row_key mapping (we
    /// emit it as a sibling JSON file via `save_mapping_to`).
    pub fn save_to(&self, path: &Path) -> Result<(), ShadowError> {
        let path_str = path.to_string_lossy().to_string();
        self.index
            .save(&path_str)
            .map_err(|e| ShadowError::Backend {
                detail: format!("usearch::save({path_str}) failed: {e}"),
            })
    }

    /// Load HNSW vectors from a previously-saved sidecar. Replaces
    /// the in-memory index. The mapping side (`load_mapping_from`)
    /// is the caller's responsibility — call them as a pair on
    /// startup.
    pub fn load_from(&self, path: &Path) -> Result<(), ShadowError> {
        let path_str = path.to_string_lossy().to_string();
        self.index
            .load(&path_str)
            .map_err(|e| ShadowError::Backend {
                detail: format!("usearch::load({path_str}) failed: {e}"),
            })?;
        // After loading, refresh the search expansion in case the
        // sidecar was saved with a different ef_search.
        self.index.change_expansion_search(HNSW_EXPANSION_SEARCH);
        Ok(())
    }

    /// Persist the doc_id ↔ row_key mapping as JSON. Sibling to
    /// the `.usearch` sidecar. Without this, a usearch reload sees
    /// the vectors but can't translate row keys back to doc ids on
    /// search.
    pub fn save_mapping_to(&self, path: &Path) -> Result<(), ShadowError> {
        let guard = self.state.read().expect("vector index lock poisoned");
        let mapping: Vec<(String, u64)> = guard
            .doc_to_key
            .iter()
            .map(|(doc_id, key)| (doc_id.clone(), *key))
            .collect();
        let payload = MappingSnapshot {
            doc_to_key: mapping,
            free_keys: guard.free_keys.clone(),
            next_key: guard.next_key,
            reserved_capacity: guard.reserved_capacity,
        };
        let bytes = serde_json::to_vec_pretty(&payload).map_err(|e| ShadowError::Backend {
            detail: format!("vector mapping JSON encode failed: {e}"),
        })?;
        std::fs::write(path, bytes).map_err(|e| ShadowError::Io {
            detail: format!("vector mapping write to {path:?} failed: {e}"),
        })
    }

    /// Restore the doc_id ↔ row_key mapping from a sibling JSON file.
    /// Pair with `load_from` on startup.
    pub fn load_mapping_from(&self, path: &Path) -> Result<(), ShadowError> {
        let bytes = std::fs::read(path).map_err(|e| ShadowError::Io {
            detail: format!("vector mapping read from {path:?} failed: {e}"),
        })?;
        let snapshot: MappingSnapshot =
            serde_json::from_slice(&bytes).map_err(|e| ShadowError::Backend {
                detail: format!("vector mapping JSON decode failed: {e}"),
            })?;
        let mut guard = self.state.write().expect("vector index lock poisoned");
        guard.doc_to_key.clear();
        guard.key_to_doc.clear();
        for (doc_id, key) in snapshot.doc_to_key {
            guard.doc_to_key.insert(doc_id.clone(), key);
            guard.key_to_doc.insert(key, doc_id);
        }
        guard.free_keys = snapshot.free_keys;
        guard.next_key = snapshot.next_key;
        guard.reserved_capacity = snapshot.reserved_capacity;
        Ok(())
    }

    /// Grow the underlying usearch reserved capacity in 2× chunks
    /// when a write would cross the boundary. usearch panics if you
    /// add past `reserve()` without growing — this guard mirrors
    /// graph-engine's production pattern.
    fn ensure_reserve(&self, key: u64) -> Result<(), ShadowError> {
        let mut guard = self.state.write().expect("vector index lock poisoned");
        if (key as usize) < guard.reserved_capacity {
            return Ok(());
        }
        let new_capacity = guard.reserved_capacity * RESERVE_GROWTH_FACTOR;
        self.index
            .reserve(new_capacity)
            .map_err(|e| ShadowError::Backend {
                detail: format!("usearch reserve(grow to {new_capacity}) failed: {e}"),
            })?;
        guard.reserved_capacity = new_capacity;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed_vec(dim: usize, fill: f32) -> Vec<f32> {
        let mut v = vec![fill; dim];
        // Tiny nudge so the cosine-norm denominator isn't zero.
        v[0] = fill + 0.001;
        v
    }

    #[test]
    fn new_with_zero_dimension_errors() {
        match VectorIndex::new(0) {
            Err(ShadowError::Backend { detail }) => {
                assert!(detail.contains("dimension must be > 0"))
            }
            Err(other) => panic!("expected Backend error; got {other:?}"),
            Ok(_) => panic!("zero-dim VectorIndex MUST be rejected"),
        }
    }

    #[test]
    fn add_then_search_returns_hit_with_perfect_similarity() {
        let idx = VectorIndex::new(8).unwrap();
        let v = fixed_vec(8, 1.0);
        idx.add("doc-A", &v).unwrap();

        let hits = idx.search(&v, 4);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, "doc-A");
        // Cosine similarity of identical vector with itself is ~1.0.
        assert!(hits[0].1 > 0.99, "got similarity {}", hits[0].1);
    }

    #[test]
    fn dimension_mismatch_returns_empty_results_not_panic() {
        let idx = VectorIndex::new(8).unwrap();
        let _ = idx.add("doc-A", &fixed_vec(8, 1.0));
        let hits = idx.search(&fixed_vec(16, 1.0), 4);
        assert!(hits.is_empty(), "wrong-dim query MUST return empty");
    }

    #[test]
    fn add_then_remove_excludes_from_search() {
        let idx = VectorIndex::new(8).unwrap();
        let v = fixed_vec(8, 1.0);
        idx.add("doc-A", &v).unwrap();
        idx.remove("doc-A").unwrap();
        let hits = idx.search(&v, 4);
        assert!(hits.is_empty(), "removed doc MUST NOT appear in search");
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn remove_unknown_is_idempotent() {
        let idx = VectorIndex::new(8).unwrap();
        // Should NOT error — caller's "remove + reinsert" pattern depends on this
        idx.remove("never-inserted").unwrap();
        idx.remove("never-inserted").unwrap();
        assert!(idx.is_empty());
    }

    #[test]
    fn add_replaces_vector_for_existing_doc_id() {
        let idx = VectorIndex::new(8).unwrap();
        idx.add("doc-A", &fixed_vec(8, 1.0)).unwrap();
        idx.add("doc-A", &fixed_vec(8, 0.5)).unwrap();
        // Still only one row in the mapping
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn search_limit_zero_returns_empty() {
        let idx = VectorIndex::new(8).unwrap();
        let v = fixed_vec(8, 1.0);
        idx.add("doc-A", &v).unwrap();
        assert!(idx.search(&v, 0).is_empty());
    }

    #[test]
    fn save_and_load_round_trips_vectors_plus_mapping() {
        let dir = tempfile::TempDir::new().unwrap();
        let usearch_path = dir.path().join("vectors.usearch");
        let mapping_path = dir.path().join("vectors.mapping.json");

        // Build + populate
        let original = VectorIndex::new(8).unwrap();
        let v1 = fixed_vec(8, 1.0);
        let v2 = fixed_vec(8, 0.5);
        original.add("doc-A", &v1).unwrap();
        original.add("doc-B", &v2).unwrap();
        original.save_to(&usearch_path).unwrap();
        original.save_mapping_to(&mapping_path).unwrap();

        // Restore into a fresh instance + verify search works
        let restored = VectorIndex::new(8).unwrap();
        restored.load_from(&usearch_path).unwrap();
        restored.load_mapping_from(&mapping_path).unwrap();
        assert_eq!(restored.len(), 2);

        let hits = restored.search(&v1, 4);
        assert_eq!(hits.len(), 2, "restored index MUST find both vectors");
        // doc-A should rank first against its own embedding
        assert_eq!(hits[0].0, "doc-A");
    }

    #[test]
    fn load_mapping_preserves_free_key_recycling() {
        let dir = tempfile::TempDir::new().unwrap();
        let usearch_path = dir.path().join("vectors.usearch");
        let mapping_path = dir.path().join("vectors.mapping.json");

        let original = VectorIndex::new(8).unwrap();
        original.add("a", &fixed_vec(8, 1.0)).unwrap();
        original.add("b", &fixed_vec(8, 0.5)).unwrap();
        original.add("c", &fixed_vec(8, 0.25)).unwrap();
        original.remove("b").unwrap(); // key 1 → free_keys
        original.save_to(&usearch_path).unwrap();
        original.save_mapping_to(&mapping_path).unwrap();

        let restored = VectorIndex::new(8).unwrap();
        restored.load_from(&usearch_path).unwrap();
        restored.load_mapping_from(&mapping_path).unwrap();

        // Insert a new doc — should reuse key 1 from the restored free list
        restored.add("d", &fixed_vec(8, 0.8)).unwrap();
        let guard = restored.state.read().unwrap();
        assert_eq!(
            guard.doc_to_key.get("d").copied(),
            Some(1),
            "free-key recycling MUST survive persistence round-trip"
        );
    }

    #[test]
    fn key_recycling_on_remove_keeps_keyspace_compact() {
        // Insert a, b, c → keys 0, 1, 2. Remove b. Insert d → should
        // reuse key 1 from the free list, NOT mint key 3.
        let idx = VectorIndex::new(8).unwrap();
        idx.add("a", &fixed_vec(8, 1.0)).unwrap();
        idx.add("b", &fixed_vec(8, 0.5)).unwrap();
        idx.add("c", &fixed_vec(8, 0.25)).unwrap();
        idx.remove("b").unwrap();
        idx.add("d", &fixed_vec(8, 0.8)).unwrap();

        let guard = idx.state.read().unwrap();
        // d's key must equal what b's was (1) — the recycled key.
        assert_eq!(guard.doc_to_key.get("d").copied(), Some(1));
        // next_key stayed at 3 — recycling kicked in
        assert_eq!(guard.next_key, 3);
    }
}
