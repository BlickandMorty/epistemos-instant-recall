//! # `epistemos-shadow`
//!
//! Contextual Shadows engine — the V1 differentiator per
//! `ambient/EPISTEMOS_V1_DECISION.md`.
//!
//! Per `docs/audits/EXTENDED_PROGRAM_PLAN_2026_04_25.md` Wave 8.1
//! (cross-ref the V1 decision: "type a sentence, see a related thought
//! appear, can't remember a time before it worked that way").
//!
//! ## Layered architecture (per epistemos_code_verdict.md)
//!
//! - **Rust here** owns CPU-bound batch compute: embedder (Model2Vec),
//!   vector index (usearch HNSW), lexical index (tantivy BM25), RRF
//!   fusion. CPU-parallel + allocator-controllable.
//! - **Swift `HaloController`** owns the @MainActor UI surface: state
//!   machine, NSPanel, SwiftUI views. Zero FFI on the typing path.
//!
//! ## V1 backend scope
//!
//! The FFI surface dispatches through `ShadowBackend`. App bootstrap
//! opens the persistent `RealBackend`; an in-memory fallback remains
//! for pre-open/test contexts and labels its hits explicitly as
//! fallback results.
//!
//! ## Performance budget (per V1 decision §"performance budget")
//!
//! | Phase                          | Target  | Hard ceiling |
//! |--------------------------------|---------|--------------|
//! | FFI hop (Swift → Rust)         | <0.5 ms | 1 ms         |
//! | Model2Vec encode (paragraph)   | <2 ms   | 4 ms         |
//! | usearch HNSW search (top-20)   | <5 ms   | 10 ms        |
//! | Tantivy BM25 search            | <8 ms   | 12 ms        |
//! | RRF fusion + metadata fetch    | <3 ms   | 5 ms         |
//! | **End-to-end recall pass**     | **<25 ms** | **40 ms** |
//!
//! All paths instrumented with the canonical Sig.* OSSignposters in
//! Wave 2.1 (subsystem `io.epistemos.core`, category `storage` for
//! the index path, `ffi` for the C ABI surface).

pub mod backend;
pub mod error;
pub mod honest_handle;
#[cfg(feature = "semantic")]
pub mod state;

pub use backend::ShadowBackend;
pub use error::ShadowError;

// ---------------------------------------------------------------------------
// Public domain types — match the Swift reference at
// ambient/HaloController.swift's ShadowDomain + ShadowHit + ShadowDocument.
// ---------------------------------------------------------------------------

/// One indexable document the controller sends to the engine.
/// Mirrors `ShadowHit` on the Swift side via the same field names.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ShadowDocument {
    pub doc_id: String,
    pub title: String,
    pub body: String,
    /// "note" | "chat" — see `ShadowDomain`. Stored as a string in the
    /// FFI surface so adding a new domain is a non-breaking change.
    pub domain: String,
    /// Sidecar metadata: which vault this doc originated from.
    /// Mirrors `GraphNodeMetadata.originVaultKey` on the Swift side
    /// (`Epistemos/Models/GraphTypes.swift`) and feeds the same lenient
    /// nil-passthrough vault filter — when None, the doc passes every
    /// vault filter (partial-rollout escape valve). `#[serde(default)]`
    /// so pre-sidecar `docs.json` snapshots reload without error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin_vault_key: Option<String>,
}

/// One result returned to the Swift controller.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ShadowHit {
    pub doc_id: String,
    pub title: String,
    /// Pre-truncated to ~160 chars by the engine so the @MainActor
    /// renderer never has to scan the full body.
    pub snippet: String,
    pub score: f32,
    /// "lexical" | "dense" | "rrf" | "in-memory-substring" — origin
    /// signal so the UI can optionally show provenance.
    pub source: String,
    /// Sidecar metadata mirrored from the source `ShadowDocument`.
    /// When the indexer populated `origin_vault_key`, the search path
    /// echoes it back so the Halo panel can render a vault-origin pill
    /// AND so the graph's lenient nil-passthrough vault filter applies
    /// uniformly across Shadow + graph node visibility.
    /// `#[serde(default)]` so pre-sidecar FFI wire format still parses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin_vault_key: Option<String>,
}

/// Aggregate index stats for the developer panel.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ShadowStats {
    pub note_count: u64,
    pub chat_count: u64,
    pub index_size_bytes: u64,
    pub last_flush_ms_ago: u64,
}

// ---------------------------------------------------------------------------
// C ABI surface
// ---------------------------------------------------------------------------
//
// Five entry points, mirroring the reference UniFFI surface in
// ambient/epistemos_shadow.rs. Each entry wraps the body in
// `std::panic::catch_unwind` so a Rust panic returns a typed error
// instead of aborting the Swift host (matches the Wave 2.4 catch_unwind
// + panic = "unwind" contract).
//
// The C ABI uses simple JSON-string-in / JSON-string-out for the
// document type. The singleton in state.rs opens the persistent
// backend during app bootstrap and falls back to an explicitly labeled
// in-memory backend only before that open succeeds.

#[cfg(feature = "semantic")]
use std::ffi::{c_char, CStr, CString};
#[cfg(feature = "semantic")]
use std::ptr;

/// Insert one document into the index. JSON-encoded `ShadowDocument`.
/// Returns 0 on success, negative ShadowError discriminant on failure.
///
/// # Safety
///
/// `doc_json` must be a valid NUL-terminated UTF-8 string.
#[cfg(feature = "semantic")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shadow_insert_json(doc_json: *const c_char) -> i32 {
    let result = std::panic::catch_unwind(|| {
        if doc_json.is_null() {
            return ShadowError::InvalidInput {
                detail: "doc_json was null".into(),
            }
            .as_code();
        }
        // SAFETY: caller contract above.
        let cstr = unsafe { CStr::from_ptr(doc_json) };
        let json = match cstr.to_str() {
            Ok(s) => s,
            Err(_) => {
                return ShadowError::InvalidInput {
                    detail: "doc_json was not valid UTF-8".into(),
                }
                .as_code();
            }
        };
        let doc: ShadowDocument = match serde_json::from_str(json) {
            Ok(d) => d,
            Err(error) => {
                return ShadowError::InvalidInput {
                    detail: format!("doc_json failed JSON parse: {error}"),
                }
                .as_code();
            }
        };
        match state::shadow_backend().insert_document(doc) {
            Ok(()) => 0,
            Err(error) => error.as_code(),
        }
    });
    result.unwrap_or(ShadowError::Panic.as_code())
}

/// Remove one document by id. Returns 0 on success.
///
/// # Safety
///
/// `doc_id` must be a valid NUL-terminated UTF-8 string.
#[cfg(feature = "semantic")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shadow_remove_json(doc_id: *const c_char) -> i32 {
    let result = std::panic::catch_unwind(|| {
        if doc_id.is_null() {
            return ShadowError::InvalidInput {
                detail: "doc_id was null".into(),
            }
            .as_code();
        }
        let cstr = unsafe { CStr::from_ptr(doc_id) };
        let id = match cstr.to_str() {
            Ok(s) => s,
            Err(_) => {
                return ShadowError::InvalidInput {
                    detail: "doc_id was not valid UTF-8".into(),
                }
                .as_code();
            }
        };
        match state::shadow_backend().remove_document(id) {
            Ok(()) => 0,
            Err(error) => error.as_code(),
        }
    });
    result.unwrap_or(ShadowError::Panic.as_code())
}

/// Search the index. Returns a JSON-encoded `Vec<ShadowHit>` as a
/// caller-owned C string. Returns null on error.
///
/// # Safety
///
/// `query` and `domain` must be valid NUL-terminated UTF-8. Caller
/// MUST pass the returned pointer to `shadow_free_string` to release
/// the allocation.
#[cfg(feature = "semantic")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shadow_search_json(
    query: *const c_char,
    domain: *const c_char,
    limit: u32,
) -> *mut c_char {
    let result = std::panic::catch_unwind(|| -> Option<CString> {
        if query.is_null() || domain.is_null() {
            return None;
        }
        let q = unsafe { CStr::from_ptr(query) }.to_str().ok()?;
        let d = unsafe { CStr::from_ptr(domain) }.to_str().ok()?;
        let hits = state::shadow_backend().search(q, d, limit as usize).ok()?;
        let json = serde_json::to_string(&hits).ok()?;
        CString::new(json).ok()
    });
    match result {
        Ok(Some(cstring)) => cstring.into_raw(),
        _ => ptr::null_mut(),
    }
}

/// Persist any pending writes to disk. Returns 0 on success.
#[cfg(feature = "semantic")]
#[unsafe(no_mangle)]
pub extern "C" fn shadow_flush() -> i32 {
    let result = std::panic::catch_unwind(|| match state::shadow_backend().flush() {
        Ok(()) => 0,
        Err(error) => error.as_code(),
    });
    result.unwrap_or(ShadowError::Panic.as_code())
}

/// Read aggregate index stats. Returns a JSON-encoded `ShadowStats`
/// as a caller-owned C string; null on error.
#[cfg(feature = "semantic")]
#[unsafe(no_mangle)]
pub extern "C" fn shadow_stats_json() -> *mut c_char {
    let result = std::panic::catch_unwind(|| -> Option<CString> {
        let stats = state::shadow_backend().stats().ok()?;
        let json = serde_json::to_string(&stats).ok()?;
        CString::new(json).ok()
    });
    match result {
        Ok(Some(cstring)) => cstring.into_raw(),
        _ => ptr::null_mut(),
    }
}

/// Free a C string returned by the FFI. Idempotent on null.
///
/// # Safety
///
/// `ptr` must come from a `*_json` function above (e.g.
/// `shadow_search_json` or `shadow_stats_json`). Calling this on any
/// other pointer is undefined behavior.
#[cfg(feature = "semantic")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shadow_free_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(|| {
        // SAFETY: caller contract.
        let _ = unsafe { CString::from_raw(ptr) };
    });
}

/// W8.4.g — initialise the global RealBackend rooted at `path`. The
/// Swift bootstrap calls this once at app start
/// (typically `~/Library/Application Support/Epistemos/shadow`).
/// Subsequent FFI calls (`shadow_insert_json`, `shadow_search_json`,
/// `shadow_flush`, `shadow_stats_json`) hit the persistent
/// RealBackend instead of the in-memory fallback. Idempotent — calling twice
/// replaces the live instance. Returns 0 on success, the
/// `ShadowError` discriminant (e.g. -4 Backend on HF download
/// failure) otherwise.
///
/// # Safety
///
/// `path` must be a valid NUL-terminated UTF-8 C string.
#[cfg(feature = "semantic")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shadow_open_at(path: *const c_char) -> i32 {
    let result = std::panic::catch_unwind(|| {
        if path.is_null() {
            return ShadowError::InvalidInput {
                detail: "path was null".into(),
            }
            .as_code();
        }
        // SAFETY: caller contract.
        let cstr = unsafe { std::ffi::CStr::from_ptr(path) };
        let path_str = match cstr.to_str() {
            Ok(s) => s,
            Err(_) => {
                return ShadowError::InvalidInput {
                    detail: "path was not valid UTF-8".into(),
                }
                .as_code();
            }
        };
        match state::open_real_backend_at(std::path::Path::new(path_str)) {
            Ok(()) => 0,
            Err(error) => error.as_code(),
        }
    });
    result.unwrap_or(ShadowError::Panic.as_code())
}
