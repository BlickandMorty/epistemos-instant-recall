//! Typed error surface for the Contextual Shadows engine.
//!
//! The C ABI returns these as small negative integers so the Swift
//! caller can `switch` on them; the discriminants are CONTRACTS
//! persisted across versions (Swift expects the same numeric meaning
//! every release).

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ShadowError {
    /// Caller passed bad input (null pointer, malformed JSON, unknown
    /// domain). Discriminant: -1.
    #[error("invalid input: {detail}")]
    InvalidInput { detail: String },

    /// Index path lookup miss. Discriminant: -2.
    #[error("document not found: {doc_id}")]
    NotFound { doc_id: String },

    /// Index file IO failure. Discriminant: -3.
    #[error("io error: {detail}")]
    Io { detail: String },

    /// Embedder / index backend internal failure. Discriminant: -4.
    #[error("backend error: {detail}")]
    Backend { detail: String },

    /// Caught a Rust panic at the FFI boundary. Discriminant: -99.
    /// Indicates a bug; Swift should log + recover.
    #[error("rust panic at FFI boundary")]
    Panic,
}

impl ShadowError {
    /// Stable numeric discriminant for the C ABI return value. Negative
    /// so callers can distinguish "non-zero error" from a successful
    /// non-zero count return. Swift mirrors these in
    /// `ShadowEngineError`.
    pub fn as_code(&self) -> i32 {
        match self {
            ShadowError::InvalidInput { .. } => -1,
            ShadowError::NotFound { .. } => -2,
            ShadowError::Io { .. } => -3,
            ShadowError::Backend { .. } => -4,
            ShadowError::Panic => -99,
        }
    }
}
