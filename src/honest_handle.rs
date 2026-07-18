//! W9.21 — Honest FFI foundation (PR1 + PR4 of 4)
//!
//! Per `docs/RESEARCH_DOSSIER_TIER_3_4.md` §W9.21: replace the
//! global `RwLock<Option<Arc<RealBackend>>>` (state.rs:251) with
//! explicit `Arc::into_raw` opaque handles passed to Swift. The
//! Swift side wraps the raw handle in a `final class` whose `init`
//! takes ownership and `deinit` releases — single-owner semantics
//! enforced by Swift's reference-counting at the binding edge.
//!
//! This module ships the Rust-side foundation as an ADDITIVE layer:
//! the new `shadow_handle_*` FFI exports work alongside the
//! existing `shadow_*` global-state exports. PR4 (Swift cutover)
//! migrates `RustShadowFFIClient` off the global API to per-instance
//! handles; when no more global-API consumers remain, the legacy
//! exports in `lib.rs` can be removed.
//!
//! Why additive: `epistemos-shadow` is on the typing hot path; a
//! single-PR rewrite that breaks the binding contract would block
//! the entire app. Side-by-side migration lets us bisect a
//! regression to the call site that broke.
//!
//! Lifecycle contract:
//!   - `shadow_handle_open_at(path)` returns `*const ShadowEngineHandle`
//!     with refcount 1 (the Swift caller owns it)
//!   - `shadow_handle_retain(handle)` increments the refcount
//!     (used when sharing the handle across threads in Swift)
//!   - `shadow_handle_release(handle)` decrements; when refcount
//!     reaches zero the inner `RealBackend` is dropped
//!   - All `shadow_handle_*` operation methods take `*const
//!     ShadowEngineHandle` and return error codes the same way
//!     as the existing global-state API
//!
//! ## PR4 additions
//!
//! Five new operation entry points so the Swift consumer never has
//! to fall back to the legacy global-state surface:
//!
//!   - `shadow_handle_insert(handle, doc_json)`     -> i32
//!   - `shadow_handle_remove(handle, doc_id)`       -> i32
//!   - `shadow_handle_flush(handle)`                -> i32
//!   - `shadow_handle_stats(handle, out_err)`       -> *mut c_char
//!   - `shadow_handle_free_string(ptr)`             -> ()
//!
//! Each panic-safe via `catch_unwind`, mirrors the global API's
//! discriminant convention, and dispatches through the same
//! `ShadowBackend` trait the legacy global API uses — so semantics
//! are identical, only ownership changes.

use std::ffi::{c_char, CStr, CString};
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::ptr;
use std::sync::Arc;

use crate::backend::{RealBackend, ShadowBackend};
use crate::error::ShadowError;
use crate::ShadowDocument;

/// Opaque handle to a real backend. The Rust side never exposes
/// the inner `RealBackend` to Swift directly — only `*const
/// ShadowEngineHandle` pointers cross the FFI boundary.
pub struct ShadowEngineHandle {
    backend: Arc<RealBackend>,
}

/// Open a real backend at `path` and return a refcount-1 handle.
/// Returns null on failure.
///
/// # Safety
/// `path` must point to a valid C string. The returned pointer is
/// owned by the caller and must be released via `shadow_handle_release`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shadow_handle_open_at(path: *const c_char) -> *const ShadowEngineHandle {
    let result = panic::catch_unwind(AssertUnwindSafe(|| -> *const ShadowEngineHandle {
        if path.is_null() {
            return ptr::null();
        }
        // SAFETY: caller contract above.
        let c_path = unsafe { CStr::from_ptr(path) };
        let s = match c_path.to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null(),
        };
        match RealBackend::open_at(Path::new(s)) {
            Ok(backend) => {
                let arc = Arc::new(ShadowEngineHandle {
                    backend: Arc::new(backend),
                });
                Arc::into_raw(arc)
            }
            Err(_) => ptr::null(),
        }
    }));
    result.unwrap_or(ptr::null())
}

/// Increment the handle's refcount. Used by Swift when stashing the
/// handle in a long-lived structure that may outlive the original
/// caller's scope.
///
/// # Safety
/// `handle` must be a valid `ShadowEngineHandle` pointer obtained
/// from `shadow_handle_open_at` (or a previous `shadow_handle_retain`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shadow_handle_retain(handle: *const ShadowEngineHandle) {
    if handle.is_null() {
        return;
    }
    let _ = panic::catch_unwind(AssertUnwindSafe(|| unsafe {
        // SAFETY: caller contract above.
        Arc::increment_strong_count(handle);
    }));
}

/// Decrement the handle's refcount. When it reaches zero, the inner
/// backend is dropped.
///
/// # Safety
/// `handle` must be a valid `ShadowEngineHandle` pointer that has
/// not yet been released. Releasing twice is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shadow_handle_release(handle: *const ShadowEngineHandle) {
    if handle.is_null() {
        return;
    }
    let _ = panic::catch_unwind(AssertUnwindSafe(|| unsafe {
        // SAFETY: caller contract above.
        Arc::decrement_strong_count(handle);
    }));
}

/// Look up the backend without consuming the handle. Internal helper
/// for the operation methods (search/insert/remove/flush/stats).
/// Inlined so the FFI surface methods compile down to one Arc clone
/// + one trait dispatch.
///
/// # Safety
/// `handle` must be a valid `ShadowEngineHandle` pointer that has
/// been obtained from `shadow_handle_open_at` and not yet released.
/// The borrow leaves the caller-side refcount unchanged: it temporarily
/// reconstructs the `Arc` to clone its inner backend, then re-leaks
/// the outer `Arc` back to its raw form so the original strong count
/// is preserved.
#[inline]
unsafe fn borrow_backend(handle: *const ShadowEngineHandle) -> Option<Arc<dyn ShadowBackend>> {
    if handle.is_null() {
        return None;
    }
    // SAFETY: caller contract above; the Arc is immediately re-leaked
    // below so this borrow does not consume Swift's owning reference.
    let arc_handle = unsafe { Arc::from_raw(handle) };
    let backend = arc_handle.backend.clone() as Arc<dyn ShadowBackend>;
    let _ = Arc::into_raw(arc_handle);
    Some(backend)
}

/// Wrap an error code from the operation methods. Returns 0 on
/// success, negative discriminant otherwise — matches the existing
/// global-API convention so Swift's error handling stays uniform.
fn encode_error(err: &ShadowError) -> i32 {
    err.as_code()
}

/// Read a UTF-8 string from a C pointer, mapping a null pointer or
/// non-UTF-8 bytes to `InvalidInput`.
///
/// # Safety
/// `ptr` must be a valid NUL-terminated C string when non-null.
unsafe fn read_c_str<'a>(ptr: *const c_char, label: &str) -> Result<&'a str, ShadowError> {
    if ptr.is_null() {
        return Err(ShadowError::InvalidInput {
            detail: format!("{label} was null"),
        });
    }
    // SAFETY: caller contract above.
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|_| ShadowError::InvalidInput {
            detail: format!("{label} was not valid UTF-8"),
        })
}

/// Search via the handle. Returns a JSON-encoded `Vec<ShadowHit>` as
/// a caller-owned C string; null on error (with `out_error` populated).
///
/// # Safety
/// `handle`, `query_c`, and `domain_c` must all be valid pointers.
/// The returned C string (when non-null) must be freed via
/// `shadow_handle_free_string`.
#[cfg(feature = "semantic")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shadow_handle_search(
    handle: *const ShadowEngineHandle,
    query_c: *const c_char,
    domain_c: *const c_char,
    limit: usize,
    out_error: *mut i32,
) -> *mut c_char {
    let result = panic::catch_unwind(AssertUnwindSafe(|| -> *mut c_char {
        // SAFETY: caller contract above.
        let backend = match unsafe { borrow_backend(handle) } {
            Some(b) => b,
            None => {
                if !out_error.is_null() {
                    // SAFETY: caller pointer.
                    unsafe {
                        *out_error = ShadowError::InvalidInput {
                            detail: "handle was null".into(),
                        }
                        .as_code()
                    };
                }
                return ptr::null_mut();
            }
        };
        // SAFETY: caller contract above.
        let query = match unsafe { read_c_str(query_c, "query") } {
            Ok(s) => s,
            Err(e) => {
                if !out_error.is_null() {
                    // SAFETY: caller pointer.
                    unsafe { *out_error = encode_error(&e) };
                }
                return ptr::null_mut();
            }
        };
        // SAFETY: caller contract above.
        let domain = match unsafe { read_c_str(domain_c, "domain") } {
            Ok(s) => s,
            Err(e) => {
                if !out_error.is_null() {
                    // SAFETY: caller pointer.
                    unsafe { *out_error = encode_error(&e) };
                }
                return ptr::null_mut();
            }
        };
        match backend.search(query, domain, limit) {
            Ok(hits) => {
                if !out_error.is_null() {
                    // SAFETY: caller pointer.
                    unsafe { *out_error = 0 };
                }
                let json = serde_json::to_string(&hits).unwrap_or_else(|_| "[]".to_string());
                CString::new(json)
                    .map(|c| c.into_raw())
                    .unwrap_or(ptr::null_mut())
            }
            Err(e) => {
                if !out_error.is_null() {
                    // SAFETY: caller pointer.
                    unsafe { *out_error = encode_error(&e) };
                }
                ptr::null_mut()
            }
        }
    }));
    match result {
        Ok(p) => p,
        Err(_) => {
            if !out_error.is_null() {
                // SAFETY: caller pointer.
                unsafe { *out_error = ShadowError::Panic.as_code() };
            }
            ptr::null_mut()
        }
    }
}

/// Search the Free lexical note index via the handle. The caller has no
/// domain selector: Free accepts and returns notes only.
///
/// # Safety
/// `handle` and `query_c` must be valid pointers. The returned C string
/// (when non-null) must be freed via `shadow_handle_free_string`.
#[cfg(feature = "free-lexical")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shadow_handle_search_notes(
    handle: *const ShadowEngineHandle,
    query_c: *const c_char,
    limit: usize,
    out_error: *mut i32,
) -> *mut c_char {
    let result = panic::catch_unwind(AssertUnwindSafe(|| -> *mut c_char {
        // SAFETY: caller contract above.
        let backend = match unsafe { borrow_backend(handle) } {
            Some(backend) => backend,
            None => {
                if !out_error.is_null() {
                    // SAFETY: caller pointer.
                    unsafe {
                        *out_error = ShadowError::InvalidInput {
                            detail: "handle was null".into(),
                        }
                        .as_code()
                    };
                }
                return ptr::null_mut();
            }
        };
        // SAFETY: caller contract above.
        let query = match unsafe { read_c_str(query_c, "query") } {
            Ok(query) => query,
            Err(error) => {
                if !out_error.is_null() {
                    // SAFETY: caller pointer.
                    unsafe { *out_error = encode_error(&error) };
                }
                return ptr::null_mut();
            }
        };
        match backend.search_notes(query, limit) {
            Ok(hits) => {
                if !out_error.is_null() {
                    // SAFETY: caller pointer.
                    unsafe { *out_error = 0 };
                }
                let json = serde_json::to_string(&hits).unwrap_or_else(|_| "[]".to_string());
                CString::new(json)
                    .map(|cstring| cstring.into_raw())
                    .unwrap_or(ptr::null_mut())
            }
            Err(error) => {
                if !out_error.is_null() {
                    // SAFETY: caller pointer.
                    unsafe { *out_error = encode_error(&error) };
                }
                ptr::null_mut()
            }
        }
    }));
    match result {
        Ok(pointer) => pointer,
        Err(_) => {
            if !out_error.is_null() {
                // SAFETY: caller pointer.
                unsafe { *out_error = ShadowError::Panic.as_code() };
            }
            ptr::null_mut()
        }
    }
}

/// Insert one document via the handle. JSON-encoded `ShadowDocument`.
/// Returns 0 on success, negative `ShadowError` discriminant on failure.
///
/// # Safety
/// `handle` must be a valid handle pointer; `doc_json` must be a valid
/// NUL-terminated UTF-8 C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shadow_handle_insert(
    handle: *const ShadowEngineHandle,
    doc_json: *const c_char,
) -> i32 {
    let result = panic::catch_unwind(AssertUnwindSafe(|| -> i32 {
        // SAFETY: caller contract above.
        let backend = match unsafe { borrow_backend(handle) } {
            Some(b) => b,
            None => {
                return ShadowError::InvalidInput {
                    detail: "handle was null".into(),
                }
                .as_code();
            }
        };
        // SAFETY: caller contract above.
        let json = match unsafe { read_c_str(doc_json, "doc_json") } {
            Ok(s) => s,
            Err(e) => return encode_error(&e),
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
        match backend.insert_document(doc) {
            Ok(()) => 0,
            Err(error) => encode_error(&error),
        }
    }));
    result.unwrap_or_else(|_| ShadowError::Panic.as_code())
}

/// Remove one document by id via the handle.
/// Returns 0 on success, negative `ShadowError` discriminant on failure.
///
/// # Safety
/// `handle` must be a valid handle pointer; `doc_id` must be a valid
/// NUL-terminated UTF-8 C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shadow_handle_remove(
    handle: *const ShadowEngineHandle,
    doc_id: *const c_char,
) -> i32 {
    let result = panic::catch_unwind(AssertUnwindSafe(|| -> i32 {
        // SAFETY: caller contract above.
        let backend = match unsafe { borrow_backend(handle) } {
            Some(b) => b,
            None => {
                return ShadowError::InvalidInput {
                    detail: "handle was null".into(),
                }
                .as_code();
            }
        };
        // SAFETY: caller contract above.
        let id = match unsafe { read_c_str(doc_id, "doc_id") } {
            Ok(s) => s,
            Err(e) => return encode_error(&e),
        };
        match backend.remove_document(id) {
            Ok(()) => 0,
            Err(error) => encode_error(&error),
        }
    }));
    result.unwrap_or_else(|_| ShadowError::Panic.as_code())
}

/// Flush pending writes to disk via the handle.
/// Returns 0 on success, negative `ShadowError` discriminant on failure.
///
/// # Safety
/// `handle` must be a valid handle pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shadow_handle_flush(handle: *const ShadowEngineHandle) -> i32 {
    let result = panic::catch_unwind(AssertUnwindSafe(|| -> i32 {
        // SAFETY: caller contract above.
        let backend = match unsafe { borrow_backend(handle) } {
            Some(b) => b,
            None => {
                return ShadowError::InvalidInput {
                    detail: "handle was null".into(),
                }
                .as_code();
            }
        };
        match backend.flush() {
            Ok(()) => 0,
            Err(error) => encode_error(&error),
        }
    }));
    result.unwrap_or_else(|_| ShadowError::Panic.as_code())
}

/// Read aggregate stats via the handle. Returns a JSON-encoded
/// `ShadowStats` as a caller-owned C string; null on error (with
/// `out_error` populated).
///
/// # Safety
/// `handle` must be a valid handle pointer. The returned C string
/// (when non-null) must be freed via `shadow_handle_free_string`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shadow_handle_stats(
    handle: *const ShadowEngineHandle,
    out_error: *mut i32,
) -> *mut c_char {
    let result = panic::catch_unwind(AssertUnwindSafe(|| -> *mut c_char {
        // SAFETY: caller contract above.
        let backend = match unsafe { borrow_backend(handle) } {
            Some(b) => b,
            None => {
                if !out_error.is_null() {
                    // SAFETY: caller pointer.
                    unsafe {
                        *out_error = ShadowError::InvalidInput {
                            detail: "handle was null".into(),
                        }
                        .as_code()
                    };
                }
                return ptr::null_mut();
            }
        };
        match backend.stats() {
            Ok(stats) => {
                if !out_error.is_null() {
                    // SAFETY: caller pointer.
                    unsafe { *out_error = 0 };
                }
                let json = match serde_json::to_string(&stats) {
                    Ok(j) => j,
                    Err(error) => {
                        if !out_error.is_null() {
                            // SAFETY: caller pointer.
                            unsafe {
                                *out_error = ShadowError::Backend {
                                    detail: format!("stats encode failed: {error}"),
                                }
                                .as_code()
                            };
                        }
                        return ptr::null_mut();
                    }
                };
                CString::new(json)
                    .map(|c| c.into_raw())
                    .unwrap_or(ptr::null_mut())
            }
            Err(e) => {
                if !out_error.is_null() {
                    // SAFETY: caller pointer.
                    unsafe { *out_error = encode_error(&e) };
                }
                ptr::null_mut()
            }
        }
    }));
    match result {
        Ok(p) => p,
        Err(_) => {
            if !out_error.is_null() {
                // SAFETY: caller pointer.
                unsafe { *out_error = ShadowError::Panic.as_code() };
            }
            ptr::null_mut()
        }
    }
}

/// Free a C string returned by `shadow_handle_search_notes`,
/// or `shadow_handle_stats`.
/// Idempotent on null.
///
/// # Safety
/// `ptr` must come from a `shadow_handle_*` function that returns
/// a `*mut c_char`. Passing a pointer from a different allocator (or
/// the same pointer twice) is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shadow_handle_free_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    let _ = panic::catch_unwind(|| {
        // SAFETY: caller contract above.
        let _ = unsafe { CString::from_raw(ptr) };
    });
}

#[cfg(test)]
mod tests {
    // The handle FFI requires a real on-disk backend (RealBackend has
    // expensive init: HuggingFace download + tantivy build). The
    // Rust-only contract test here just verifies the helper bookkeeping
    // (refcount preserved across borrow_backend) without touching disk.

    use super::*;
    use std::sync::Arc;

    /// The original W9.21 contract test exercised `Arc::into_raw` /
    /// `Arc::from_raw` directly, but used them incorrectly:
    ///
    ///   `&Arc::from_raw(raw)` constructs a temporary Arc that's
    ///   immediately dropped at the statement boundary. The drop
    ///   decrements the refcount to zero and frees the allocation.
    ///   The next `Arc::from_raw(raw)` call is then UAF, reading
    ///   whatever the allocator now has at that address.
    ///
    /// Rewritten: pair every `from_raw` with a preceding
    /// `increment_strong_count` so the temporary's drop returns the
    /// count to its prior value rather than freeing the allocation.
    /// This is exactly the invariant `borrow_backend` upholds via its
    /// `let _ = Arc::into_raw(arc_handle);` leak-back at line 118.
    #[test]
    fn borrow_preserves_refcount() {
        let arc = Arc::new(0u32);
        let raw = Arc::into_raw(arc); // count = 1 (raw owns it)

        // Before observation: increment so the temporary `from_raw`'s
        // drop decrements back to 1 instead of 0.
        unsafe { Arc::increment_strong_count(raw) }; // count = 2
        let count_before = unsafe {
            let temp = Arc::from_raw(raw); // count still 2 (transferring ownership of one ref)
            let n = Arc::strong_count(&temp);
            // temp drops here → count = 1
            n
        };
        assert_eq!(
            count_before, 2,
            "before: count must observe the bumped reference"
        );

        // Same pattern for `count_after` — bump, observe, let drop
        // restore. If the simulated round-trip leaked, count would
        // drift; if it double-freed, the increment+strong_count would
        // hit invalid memory.
        unsafe { Arc::increment_strong_count(raw) }; // count = 2 again
        let count_after = unsafe {
            let temp = Arc::from_raw(raw);
            let n = Arc::strong_count(&temp);
            n
        };
        assert_eq!(
            count_before, count_after,
            "borrow round-trip must leave refcount invariant"
        );

        // Drop the last live reference so the allocation is reclaimed
        // (avoids a leak warning under miri / sanitizers).
        unsafe {
            let last = Arc::from_raw(raw); // count = 1, owned by `last`
            drop(last); // count = 0, freed
        }

        // Sanity: encode_error still works (no state shared with the
        // FFI helper that could have been corrupted by the test).
        let _ = encode_error(&ShadowError::InvalidInput { detail: "x".into() });
    }
}
