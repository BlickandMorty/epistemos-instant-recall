//! Free V1 deterministic Contextual Shadows backend.
//!
//! This implementation deliberately owns only persisted note documents and a
//! Tantivy/BM25 index. It never opens vault source files; migration works only
//! on the derived Shadow directory and is therefore safe for archived chats.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, RwLock};
use std::time::Instant;

use crate::backend::lexical_index::LexicalIndex;
use crate::backend::ShadowBackend;
use crate::error::ShadowError;
use crate::{ShadowDocument, ShadowHit, ShadowStats};

const CACHE_MARKER: &str = ".free-lexical-v1";
const MAX_DERIVED_DOCS_BYTES: u64 = 32 * 1024 * 1024;
const MAX_DERIVED_DOCUMENTS: usize = 10_000;

pub struct RealBackend {
    lexical: LexicalIndex,
    docs: RwLock<HashMap<String, ShadowDocument>>,
    last_flush: Mutex<Instant>,
    persistence_root: Option<PathBuf>,
}

impl RealBackend {
    pub fn new() -> Result<Self, ShadowError> {
        Ok(Self {
            lexical: LexicalIndex::new()?,
            docs: RwLock::new(HashMap::new()),
            last_flush: Mutex::new(Instant::now()),
            persistence_root: None,
        })
    }

    pub fn open_at(path: &Path) -> Result<Self, ShadowError> {
        std::fs::create_dir_all(path).map_err(|error| ShadowError::Io {
            detail: format!("create_dir_all({path:?}) failed: {error}"),
        })?;
        Self::reset_oversized_derived_cache(path)?;

        let lexical = LexicalIndex::open_at(&path.join("tantivy"))?;
        let mut docs = Self::load_bounded_note_documents(path)?;
        Self::purge_non_note_derived_state(path, &lexical, &mut docs)?;

        Ok(Self {
            lexical,
            docs: RwLock::new(docs),
            last_flush: Mutex::new(Instant::now()),
            persistence_root: Some(path.to_path_buf()),
        })
    }

    fn reset_oversized_derived_cache(path: &Path) -> Result<(), ShadowError> {
        let docs_path = path.join("docs.json");
        let is_oversized = docs_path
            .metadata()
            .map(|metadata| metadata.len() > MAX_DERIVED_DOCS_BYTES)
            .unwrap_or(false);
        if !is_oversized {
            return Ok(());
        }

        for derived_path in [path.join("tantivy"), path.join("vectors")] {
            if derived_path.exists() {
                std::fs::remove_dir_all(&derived_path).map_err(|error| ShadowError::Io {
                    detail: format!("remove_dir_all({derived_path:?}) failed: {error}"),
                })?;
            }
        }
        std::fs::remove_file(&docs_path).map_err(|error| ShadowError::Io {
            detail: format!("remove_file({docs_path:?}) failed: {error}"),
        })?;
        Ok(())
    }

    fn load_bounded_note_documents(
        path: &Path,
    ) -> Result<HashMap<String, ShadowDocument>, ShadowError> {
        let docs_path = path.join("docs.json");
        if !docs_path.exists() {
            return Ok(HashMap::new());
        }
        let bytes = std::fs::read(&docs_path).map_err(|error| ShadowError::Io {
            detail: format!("read({docs_path:?}) failed: {error}"),
        })?;
        let mut docs: HashMap<String, ShadowDocument> =
            serde_json::from_slice(&bytes).map_err(|error| ShadowError::Backend {
                detail: format!("docs.json decode failed: {error}"),
            })?;
        if docs.len() > MAX_DERIVED_DOCUMENTS {
            std::fs::remove_file(&docs_path).map_err(|error| ShadowError::Io {
                detail: format!("remove_file({docs_path:?}) failed: {error}"),
            })?;
            return Ok(HashMap::new());
        }
        docs.retain(|_, document| document.domain == "note");
        Ok(docs)
    }

    /// Removes every unreachable/non-note derived record without reading user
    /// vault content. Reinserted note rows make a stale lexical sidecar and
    /// `docs.json` converge before searches are exposed.
    fn purge_non_note_derived_state(
        path: &Path,
        lexical: &LexicalIndex,
        docs: &mut HashMap<String, ShadowDocument>,
    ) -> Result<(), ShadowError> {
        docs.retain(|_, document| document.domain == "note");

        let stale_ids: Vec<String> = lexical
            .iter_doc_ids()?
            .into_iter()
            .filter_map(|(doc_id, domain)| {
                (domain != "note" || !docs.contains_key(&doc_id)).then_some(doc_id)
            })
            .collect();
        for doc_id in stale_ids {
            lexical.remove(&doc_id)?;
        }
        for document in docs.values() {
            lexical.insert(document)?;
        }

        let vectors_path = path.join("vectors");
        if vectors_path.exists() {
            std::fs::remove_dir_all(&vectors_path).map_err(|error| ShadowError::Io {
                detail: format!("remove_dir_all({vectors_path:?}) failed: {error}"),
            })?;
        }
        Self::write_cache_marker(path)
    }

    fn write_cache_marker(path: &Path) -> Result<(), ShadowError> {
        let marker = path.join(CACHE_MARKER);
        let temporary = path.join(".free-lexical-v1.tmp");
        std::fs::write(&temporary, b"version=1\nnotes-only=true\n").map_err(|error| {
            ShadowError::Io {
                detail: format!("write({temporary:?}) failed: {error}"),
            }
        })?;
        std::fs::rename(&temporary, &marker).map_err(|error| ShadowError::Io {
            detail: format!("rename({temporary:?}, {marker:?}) failed: {error}"),
        })
    }

    fn persist_documents(&self, root: &Path) -> Result<(), ShadowError> {
        let docs_path = root.join("docs.json");
        let temporary = root.join("docs.json.free-lexical.tmp");
        let docs = self.docs.read().expect("docs lock poisoned");
        let bytes = serde_json::to_vec(&*docs).map_err(|error| ShadowError::Backend {
            detail: format!("docs.json encode failed: {error}"),
        })?;
        std::fs::write(&temporary, bytes).map_err(|error| ShadowError::Io {
            detail: format!("write({temporary:?}) failed: {error}"),
        })?;
        std::fs::rename(&temporary, &docs_path).map_err(|error| ShadowError::Io {
            detail: format!("rename({temporary:?}, {docs_path:?}) failed: {error}"),
        })
    }
}

impl ShadowBackend for RealBackend {
    fn insert_document(&self, document: ShadowDocument) -> Result<(), ShadowError> {
        if document.doc_id.is_empty() {
            return Err(ShadowError::InvalidInput {
                detail: "doc_id was empty".into(),
            });
        }
        if document.domain != "note" {
            return Err(ShadowError::InvalidInput {
                detail: "Free lexical backend accepts note documents only".into(),
            });
        }
        self.lexical.insert(&document)?;
        self.docs
            .write()
            .expect("docs lock poisoned")
            .insert(document.doc_id.clone(), document);
        Ok(())
    }

    fn remove_document(&self, doc_id: &str) -> Result<(), ShadowError> {
        let removed = self
            .docs
            .write()
            .expect("docs lock poisoned")
            .remove(doc_id);
        if removed.is_none() {
            return Err(ShadowError::NotFound {
                doc_id: doc_id.to_string(),
            });
        }
        self.lexical.remove(doc_id)
    }

    fn search_notes(&self, query: &str, limit: usize) -> Result<Vec<ShadowHit>, ShadowError> {
        if query.trim().is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let docs = self.docs.read().expect("docs lock poisoned");
        Ok(self
            .lexical
            .search(query, "note", limit)?
            .into_iter()
            .filter_map(|hit| {
                let document = docs.get(&hit.doc_id)?;
                Some(ShadowHit {
                    doc_id: document.doc_id.clone(),
                    title: document.title.clone(),
                    snippet: snippet(&document.body, query),
                    score: hit.score,
                    source: "lexical".into(),
                    origin_vault_key: document.origin_vault_key.clone(),
                })
            })
            .collect())
    }

    fn flush(&self) -> Result<(), ShadowError> {
        if let Some(root) = self.persistence_root.as_ref() {
            self.persist_documents(root)?;
        }
        *self.last_flush.lock().expect("flush lock poisoned") = Instant::now();
        Ok(())
    }

    fn stats(&self) -> Result<ShadowStats, ShadowError> {
        let docs = self.docs.read().expect("docs lock poisoned");
        let bytes = docs
            .values()
            .map(|document| (document.title.len() + document.body.len()) as u64)
            .sum();
        Ok(ShadowStats {
            note_count: docs.len() as u64,
            // Raw DTO compatibility only; Free has no operational chat index.
            chat_count: 0,
            index_size_bytes: bytes,
            last_flush_ms_ago: self
                .last_flush
                .lock()
                .expect("flush lock poisoned")
                .elapsed()
                .as_millis() as u64,
        })
    }
}

fn snippet(body: &str, query: &str) -> String {
    const MAX_BYTES: usize = 160;
    if body.len() <= MAX_BYTES {
        return body.to_string();
    }
    let lower_body = body.to_lowercase();
    let lower_query = query.to_lowercase();
    let center = lower_body.find(&lower_query).unwrap_or(0);
    let start = center.saturating_sub(MAX_BYTES / 2);
    let end = (start + MAX_BYTES).min(body.len());
    let safe_start = (0..=start)
        .rev()
        .find(|index| body.is_char_boundary(*index))
        .unwrap_or(0);
    let safe_end = (end..=body.len())
        .find(|index| body.is_char_boundary(*index))
        .unwrap_or(body.len());
    body[safe_start..safe_end].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexical_backend_indexes_notes_and_rejects_other_domains() {
        let backend = RealBackend::new().unwrap();
        backend
            .insert_document(ShadowDocument {
                doc_id: "note-1".into(),
                title: "Quarterly report".into(),
                body: "Revenue grew across all regions.".into(),
                domain: "note".into(),
                origin_vault_key: None,
            })
            .unwrap();

        let hits = backend.search_notes("revenue", 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].source, "lexical");
        assert!(backend
            .insert_document(ShadowDocument {
                doc_id: "legacy".into(),
                title: "Archived".into(),
                body: "Never index this in Free.".into(),
                domain: "chat".into(),
                origin_vault_key: None,
            })
            .is_err());
    }
}
