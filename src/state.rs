//! Global singleton state for the Shadow engine — in-memory fallback backend.
//!
//! The app opens `RealBackend` during bootstrap. This in-memory
//! HashMap backend remains only as a pre-open/test fallback so the
//! FFI surface fails honestly and keeps deterministic contract tests.
//!
//! The singleton pattern matches the reference at
//! `ambient/epistemos_shadow.rs` — exactly one engine instance per
//! process, reachable through `shadow_state()`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use parking_lot::RwLock;

use crate::backend::{RealBackend, ShadowBackend};
use crate::error::ShadowError;
use crate::{ShadowDocument, ShadowHit, ShadowStats};

#[cfg(feature = "free-lexical")]
fn accepts_domain(domain: &str) -> bool {
    domain == "note"
}

#[cfg(feature = "semantic")]
fn accepts_domain(domain: &str) -> bool {
    domain == "note" || domain == "chat"
}

impl Default for ShadowState {
    /// Build a fresh in-memory backend. The trait-conformance test in
    /// `backend::tests` uses this to verify the W8.4.a contract; the
    /// production singleton at `shadow_state()` still goes through
    /// `ShadowState::new()`.
    fn default() -> Self {
        Self::new()
    }
}

/// `ShadowState` satisfies the `ShadowBackend` trait so the FFI
/// surface can dispatch through one trait object for both the real
/// backend and this in-memory fallback.
impl ShadowBackend for ShadowState {
    fn insert_document(&self, doc: ShadowDocument) -> Result<(), ShadowError> {
        ShadowState::insert_document(self, doc)
    }

    fn remove_document(&self, doc_id: &str) -> Result<(), ShadowError> {
        ShadowState::remove_document(self, doc_id)
    }

    fn search(
        &self,
        query: &str,
        domain: &str,
        limit: usize,
    ) -> Result<Vec<ShadowHit>, ShadowError> {
        ShadowState::search(self, query, domain, limit)
    }

    fn flush(&self) -> Result<(), ShadowError> {
        ShadowState::flush(self)
    }

    fn stats(&self) -> Result<ShadowStats, ShadowError> {
        ShadowState::stats(self)
    }
}

/// In-memory fallback backend — a HashMap of doc_id → document.
/// Search is a naive substring scan, scored by snippet position.
/// Sufficient for HaloController state-machine tests and pre-open FFI
/// contract verification; production bootstrap opens `RealBackend`.
pub struct ShadowState {
    inner: RwLock<Inner>,
}

struct Inner {
    docs: HashMap<String, ShadowDocument>,
    last_flush: Instant,
}

impl Default for Inner {
    fn default() -> Self {
        Self {
            docs: HashMap::new(),
            last_flush: Instant::now(),
        }
    }
}

impl ShadowState {
    fn new() -> Self {
        Self {
            inner: RwLock::new(Inner::default()),
        }
    }

    /// Insert or replace a document.
    pub fn insert_document(&self, doc: ShadowDocument) -> Result<(), ShadowError> {
        if doc.doc_id.is_empty() {
            return Err(ShadowError::InvalidInput {
                detail: "doc_id was empty".into(),
            });
        }
        if !accepts_domain(&doc.domain) {
            return Err(ShadowError::InvalidInput {
                detail: format!("unsupported domain '{}'", doc.domain),
            });
        }
        let mut guard = self.inner.write();
        guard.docs.insert(doc.doc_id.clone(), doc);
        Ok(())
    }

    /// Remove a document. Idempotent — removing an unknown doc returns NotFound.
    pub fn remove_document(&self, doc_id: &str) -> Result<(), ShadowError> {
        let mut guard = self.inner.write();
        if guard.docs.remove(doc_id).is_some() {
            Ok(())
        } else {
            Err(ShadowError::NotFound {
                doc_id: doc_id.to_string(),
            })
        }
    }

    /// Search the index. The fallback does case-insensitive substring
    /// matching across title + body and ranks by:
    ///   - title match (score += 2.0)
    ///   - body match position (closer to start scores higher)
    ///   - exact-token match (score += 0.5)
    ///
    /// The real backend uses Model2Vec + usearch HNSW + tantivy BM25
    /// + RRF fusion per the V1 decision.
    pub fn search(
        &self,
        query: &str,
        domain: &str,
        limit: usize,
    ) -> Result<Vec<ShadowHit>, ShadowError> {
        if !accepts_domain(domain) {
            return Err(ShadowError::InvalidInput {
                detail: format!("unsupported search domain '{domain}'"),
            });
        }
        let query_lower = query.to_lowercase();
        if query_lower.trim().is_empty() {
            return Ok(Vec::new());
        }

        let guard = self.inner.read();
        let mut hits: Vec<ShadowHit> = guard
            .docs
            .values()
            .filter(|doc| doc.domain == domain)
            .filter_map(|doc| {
                let title_lower = doc.title.to_lowercase();
                let body_lower = doc.body.to_lowercase();
                let title_hit = title_lower.contains(&query_lower);
                let body_hit_pos = body_lower.find(&query_lower);
                if !title_hit && body_hit_pos.is_none() {
                    return None;
                }

                let mut score: f32 = 0.0;
                if title_hit {
                    score += 2.0;
                }
                if let Some(pos) = body_hit_pos {
                    let body_len = doc.body.len().max(1) as f32;
                    score += 1.0 - (pos as f32 / body_len);
                }
                if title_lower.split_whitespace().any(|tok| tok == query_lower) {
                    score += 0.5;
                }

                let snippet = build_snippet(&doc.body, body_hit_pos);
                Some(ShadowHit {
                    doc_id: doc.doc_id.clone(),
                    title: doc.title.clone(),
                    snippet,
                    score,
                    source: "in-memory-substring".to_string(),
                    origin_vault_key: doc.origin_vault_key.clone(),
                })
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(limit);
        Ok(hits)
    }

    /// Persist the index. The fallback records the flush instant for
    /// the stats endpoint; `RealBackend` owns disk persistence.
    pub fn flush(&self) -> Result<(), ShadowError> {
        let mut guard = self.inner.write();
        guard.last_flush = Instant::now();
        Ok(())
    }

    /// Aggregate stats for the developer panel.
    pub fn stats(&self) -> Result<ShadowStats, ShadowError> {
        let guard = self.inner.read();
        let mut note_count: u64 = 0;
        let mut chat_count: u64 = 0;
        let mut bytes: u64 = 0;
        for doc in guard.docs.values() {
            bytes += (doc.title.len() + doc.body.len()) as u64;
            match doc.domain.as_str() {
                "note" => note_count += 1,
                "chat" => chat_count += 1,
                _ => {}
            }
        }
        Ok(ShadowStats {
            note_count,
            chat_count,
            index_size_bytes: bytes,
            last_flush_ms_ago: guard.last_flush.elapsed().as_millis() as u64,
        })
    }
}

static SHADOW_STATE: OnceLock<ShadowState> = OnceLock::new();

/// Process-wide engine handle. Lazy-initialised on first call so an
/// app that never uses Contextual Shadows pays zero startup cost.
pub fn shadow_state() -> &'static ShadowState {
    SHADOW_STATE.get_or_init(ShadowState::new)
}

// MARK: - W8.4.g — pluggable backend singleton
//
// Two-tier dispatch:
//   1. If `shadow_open_at(path)` has been called successfully, every
//      `shadow_backend()` call returns the live `RealBackend` (the
//      W8.4.e production stack).
//   2. Otherwise, returns the in-memory fallback via `shadow_state()`
//      upcast to the trait — preserves deterministic pre-open FFI
//      behavior without claiming persistent search.
//
// The runtime upgrade is the entire point: callers (the FFI surface
// in lib.rs) ALWAYS dispatch through `shadow_backend()`. The Swift
// bootstrap fires `shadow_open_at(path)` once at app start; all
// subsequent FFI calls hit the persistent RealBackend.

static REAL_BACKEND: RwLock<Option<Arc<RealBackend>>> = RwLock::new(None);

/// Initialise the global RealBackend at `path`. Idempotent — calling
/// twice with the same (or different) path replaces the live
/// instance. Returns the ShadowError discriminant on failure (e.g.
/// HuggingFace download failed when offline + cold cache).
pub fn open_real_backend_at(path: &Path) -> Result<(), ShadowError> {
    let backend = RealBackend::open_at(path)?;
    let mut guard = REAL_BACKEND.write();
    *guard = Some(Arc::new(backend));
    Ok(())
}

/// Process-wide ShadowBackend handle. Returns `RealBackend` when
/// `open_real_backend_at` has been called; otherwise the in-memory
/// fallback (lazily-built singleton — pre-open inserts share state
/// across calls, just like the original
/// `shadow_state()` semantics).
pub fn shadow_backend() -> Arc<dyn ShadowBackend> {
    if let Some(real) = REAL_BACKEND.read().clone() {
        return real;
    }
    static IN_MEMORY_FALLBACK: OnceLock<Arc<ShadowState>> = OnceLock::new();
    IN_MEMORY_FALLBACK
        .get_or_init(|| Arc::new(ShadowState::new()))
        .clone()
}

/// Reset the global RealBackend (test-only). Releases the Arc so
/// the next `shadow_backend()` call falls back to the in-memory backend. Used
/// by the W8.4.g singleton-flip tests to give each test a fresh
/// world without process-restart.
#[cfg(test)]
pub fn _reset_real_backend_for_tests() {
    *REAL_BACKEND.write() = None;
}

/// Pre-truncated snippet centred around the body match (or the
/// document head when only the title matched). Capped at 160 chars
/// per the V1 decision §"shadow hit shape".
fn build_snippet(body: &str, hit_pos: Option<usize>) -> String {
    const MAX: usize = 160;
    if body.len() <= MAX {
        return body.to_string();
    }
    let center = hit_pos.unwrap_or(0);
    let half = MAX / 2;
    let start = center.saturating_sub(half);
    let end = (start + MAX).min(body.len());
    // Snap to char boundaries so we don't slice mid-codepoint.
    let safe_start = (0..=start)
        .rev()
        .find(|i| body.is_char_boundary(*i))
        .unwrap_or(0);
    let safe_end = (end..=body.len())
        .find(|i| body.is_char_boundary(*i))
        .unwrap_or(body.len());
    body[safe_start..safe_end].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    // MARK: - W8.4.g singleton flip

    #[test]
    fn shadow_backend_falls_back_to_in_memory_when_no_real_backend_open() {
        // Reset to ensure a clean world (other tests may have flipped it).
        _reset_real_backend_for_tests();
        let backend = shadow_backend();
        // Fallback semantics: empty doc_id rejected on insert.
        let result = backend.insert_document(ShadowDocument {
            doc_id: "".to_string(),
            domain: "note".to_string(),
            title: "x".to_string(),
            body: "y".to_string(),
            origin_vault_key: None,
        });
        assert!(matches!(result, Err(ShadowError::InvalidInput { .. })));
    }

    #[test]
    fn shadow_backend_in_memory_fallback_shares_state_across_calls() {
        _reset_real_backend_for_tests();
        let a = shadow_backend();
        let b = shadow_backend();
        // Insert through `a`, search through `b` — must hit the same
        // IN_MEMORY_FALLBACK singleton.
        a.insert_document(ShadowDocument {
            doc_id: "share-test".to_string(),
            domain: "note".to_string(),
            title: "shared".to_string(),
            body: "body shared body".to_string(),
            origin_vault_key: None,
        })
        .unwrap();
        let hits = b.search("shared", "note", 5).unwrap();
        assert!(
            hits.iter().any(|h| h.doc_id == "share-test"),
            "the second shadow_backend() handle MUST see the first one's writes"
        );
    }

    fn fresh_state() -> ShadowState {
        ShadowState::new()
    }

    fn note(id: &str, title: &str, body: &str) -> ShadowDocument {
        ShadowDocument {
            doc_id: id.into(),
            title: title.into(),
            body: body.into(),
            domain: "note".into(),
            origin_vault_key: None,
        }
    }

    #[test]
    fn insert_then_search_returns_hit() {
        let state = fresh_state();
        state
            .insert_document(note(
                "n1",
                "Kant on duty",
                "Categorical imperative discussion",
            ))
            .unwrap();
        let hits = state.search("kant", "note", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "n1");
        assert!(hits[0].score > 0.0);
        assert_eq!(hits[0].source, "in-memory-substring");
    }

    #[test]
    fn origin_vault_key_round_trips_through_search() {
        // Sidecar metadata B.x — `origin_vault_key` set on a
        // ShadowDocument must echo back on the matching ShadowHit so the
        // host can apply the same lenient nil-passthrough vault filter
        // the graph already uses.
        let state = fresh_state();
        state
            .insert_document(ShadowDocument {
                doc_id: "n-vk".into(),
                title: "Kant on duty".into(),
                body: "Categorical imperative discussion".into(),
                domain: "note".into(),
                origin_vault_key: Some("vault-alpha".into()),
            })
            .unwrap();
        let hits = state.search("kant", "note", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].origin_vault_key.as_deref(), Some("vault-alpha"));
    }

    #[test]
    fn origin_vault_key_nil_passthrough_preserved() {
        // Docs inserted without a vault key emit hits with None — the
        // lenient nil-passthrough contract documented on GraphNodeMetadata.
        let state = fresh_state();
        state
            .insert_document(note("n-nil", "Title", "body"))
            .unwrap();
        let hits = state.search("body", "note", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].origin_vault_key.is_none());
    }

    #[test]
    fn empty_query_returns_empty_results() {
        let state = fresh_state();
        state
            .insert_document(note("n1", "anything", "anything"))
            .unwrap();
        assert!(state.search("", "note", 10).unwrap().is_empty());
        assert!(state.search("   ", "note", 10).unwrap().is_empty());
    }

    #[test]
    fn shadow_document_omits_origin_vault_key_when_nil() {
        // Mirror of the Swift `shadowDocumentDTOOmitsNilOriginVaultKey`
        // test in EpistemosTests/ShadowServicesTests.swift. Pre-sidecar
        // consumers (docs.json snapshots minted before 2026-05-15) must
        // see byte-identical document JSON; that's enforced by the
        // `#[serde(skip_serializing_if = "Option::is_none")]` attribute
        // on `ShadowDocument.origin_vault_key`. A future PR that drops
        // the skip-empty attribute would silently emit
        // `"origin_vault_key":null` in every doc JSON.
        let doc = ShadowDocument {
            doc_id: "n1".into(),
            title: "Kant on duty".into(),
            body: "Categorical imperative".into(),
            domain: "note".into(),
            origin_vault_key: None,
        };
        let json = serde_json::to_string(&doc).unwrap();
        assert!(
            !json.contains("origin_vault_key"),
            "nil origin_vault_key MUST NOT serialize; got: {json}"
        );
    }

    #[test]
    fn shadow_document_round_trips_origin_vault_key_when_set() {
        // Populated case: encode → decode round-trip preserves the
        // field value. Mirrors Swift's `shadowDocumentDTORoundTrips
        // OriginVaultKey`.
        let doc = ShadowDocument {
            doc_id: "n2".into(),
            title: "vault-alpha note".into(),
            body: "body".into(),
            domain: "note".into(),
            origin_vault_key: Some("vault-alpha".into()),
        };
        let json = serde_json::to_string(&doc).unwrap();
        assert!(json.contains("\"origin_vault_key\":\"vault-alpha\""));
        let decoded: ShadowDocument = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.origin_vault_key.as_deref(), Some("vault-alpha"));
        assert_eq!(decoded.doc_id, "n2");
    }

    #[test]
    fn shadow_document_decodes_pre_sidecar_json_without_field() {
        // Back-compat: docs.json from before 2026-05-15 has no
        // `origin_vault_key` field at all. The `#[serde(default)]`
        // attribute makes the missing field decode to None. Mirrors
        // Swift's `shadowDocumentDTODecodesPreSidecarJSON`.
        let pre_sidecar = r#"{"doc_id":"old","title":"T","body":"B","domain":"note"}"#;
        let decoded: ShadowDocument = serde_json::from_str(pre_sidecar).unwrap();
        assert!(decoded.origin_vault_key.is_none());
        assert_eq!(decoded.doc_id, "old");
    }

    #[test]
    fn unknown_domain_rejected_on_search() {
        let state = fresh_state();
        let err = state.search("kant", "unknown", 10).unwrap_err();
        assert!(matches!(err, ShadowError::InvalidInput { .. }));
    }

    #[test]
    fn unknown_domain_rejected_on_insert() {
        let state = fresh_state();
        let err = state
            .insert_document(ShadowDocument {
                doc_id: "x".into(),
                title: "x".into(),
                body: "x".into(),
                domain: "unknown".into(),
                origin_vault_key: None,
            })
            .unwrap_err();
        assert!(matches!(err, ShadowError::InvalidInput { .. }));
    }

    #[test]
    fn empty_doc_id_rejected_on_insert() {
        let state = fresh_state();
        let err = state
            .insert_document(ShadowDocument {
                doc_id: "".into(),
                title: "t".into(),
                body: "b".into(),
                domain: "note".into(),
                origin_vault_key: None,
            })
            .unwrap_err();
        assert!(matches!(err, ShadowError::InvalidInput { .. }));
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn search_filters_by_domain() {
        let state = fresh_state();
        state
            .insert_document(note("n1", "kant", "kant body"))
            .unwrap();
        state
            .insert_document(ShadowDocument {
                doc_id: "c1".into(),
                title: "kant chat".into(),
                body: "kant chat body".into(),
                domain: "chat".into(),
                origin_vault_key: None,
            })
            .unwrap();
        let notes = state.search("kant", "note", 10).unwrap();
        let chats = state.search("kant", "chat", 10).unwrap();
        assert_eq!(notes.len(), 1);
        assert_eq!(chats.len(), 1);
        assert_eq!(notes[0].doc_id, "n1");
        assert_eq!(chats[0].doc_id, "c1");
    }

    #[test]
    fn title_hit_outranks_body_only_hit() {
        let state = fresh_state();
        state
            .insert_document(note("title", "kant on duty", "unrelated body"))
            .unwrap();
        state
            .insert_document(note(
                "body",
                "unrelated title",
                "this mentions kant in passing",
            ))
            .unwrap();
        let hits = state.search("kant", "note", 10).unwrap();
        assert_eq!(
            hits[0].doc_id, "title",
            "title match must outrank body-only match"
        );
    }

    #[test]
    fn limit_caps_results() {
        let state = fresh_state();
        for i in 0..20 {
            state
                .insert_document(note(&format!("n{i}"), "kant", "kant"))
                .unwrap();
        }
        let hits = state.search("kant", "note", 5).unwrap();
        assert_eq!(hits.len(), 5);
    }

    #[test]
    fn remove_then_search_returns_empty() {
        let state = fresh_state();
        state.insert_document(note("n1", "kant", "kant")).unwrap();
        state.remove_document("n1").unwrap();
        assert!(state.search("kant", "note", 10).unwrap().is_empty());
    }

    #[test]
    fn remove_unknown_returns_not_found() {
        let state = fresh_state();
        let err = state.remove_document("unknown").unwrap_err();
        assert!(matches!(err, ShadowError::NotFound { .. }));
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn stats_track_counts_per_domain() {
        let state = fresh_state();
        state.insert_document(note("n1", "a", "b")).unwrap();
        state.insert_document(note("n2", "c", "d")).unwrap();
        state
            .insert_document(ShadowDocument {
                doc_id: "c1".into(),
                title: "x".into(),
                body: "y".into(),
                domain: "chat".into(),
                origin_vault_key: None,
            })
            .unwrap();
        let stats = state.stats().unwrap();
        assert_eq!(stats.note_count, 2);
        assert_eq!(stats.chat_count, 1);
        assert!(stats.index_size_bytes > 0);
    }

    #[test]
    fn snippet_handles_unicode_boundary() {
        // 200-char body with multibyte chars; snippet must not slice
        // mid-codepoint.
        let mut body = String::new();
        for _ in 0..40 {
            body.push_str("hellö "); // ö is 2 bytes in UTF-8
        }
        let state = fresh_state();
        state.insert_document(note("u", "title", &body)).unwrap();
        let hits = state.search("hellö", "note", 1).unwrap();
        // The snippet is valid UTF-8 (didn't panic on slice).
        assert!(!hits.is_empty());
        let _ = hits[0].snippet.chars().count();
    }
}
