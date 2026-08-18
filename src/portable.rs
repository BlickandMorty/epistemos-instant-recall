//! Cross-platform facade for Epistemos's two retrieval experiences.
//!
//! `search_sidebar` is explicit, query-driven search. `recall_ambient` is
//! intentionally separate: it accepts the text currently being written and
//! excludes the origin note. Both use one local index and fuse title, BM25,
//! and deterministic hashed-trigram vector rankings with reciprocal rank
//! fusion (RRF). No network or model download is required.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use crate::backend::{RealBackend, ShadowBackend};
use crate::{ShadowDocument, ShadowError};

const CATALOG: &str = "portable-catalog.json";
const VECTOR_DIM: usize = 1024;
const RRF_K: f32 = 60.0;
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RecallMode {
    Sidebar,
    Ambient,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecallResult {
    pub doc_id: String,
    pub title: String,
    pub snippet: String,
    pub score: f32,
    pub channels: Vec<String>,
    pub mode: RecallMode,
    pub elapsed_us: u128,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IndexReport {
    pub indexed: usize,
    pub skipped: usize,
    pub failed: usize,
}

pub struct RecallEngine {
    root: PathBuf,
    lexical: RealBackend,
    docs: HashMap<String, ShadowDocument>,
}

impl RecallEngine {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, ShadowError> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(io_error)?;
        let docs = load_catalog(&root.join(CATALOG))?;
        let lexical = RealBackend::open_at(&root.join("engine"))?;
        Ok(Self {
            root,
            lexical,
            docs,
        })
    }

    pub fn upsert(&mut self, document: ShadowDocument) -> Result<(), ShadowError> {
        self.lexical.insert_document(document.clone())?;
        self.docs.insert(document.doc_id.clone(), document);
        Ok(())
    }

    pub fn flush(&self) -> Result<(), ShadowError> {
        self.lexical.flush()?;
        let bytes = serde_json::to_vec_pretty(&self.docs).map_err(backend_error)?;
        let target = self.root.join(CATALOG);
        let temporary = self.root.join(".portable-catalog.tmp");
        fs::write(&temporary, bytes).map_err(io_error)?;
        fs::rename(&temporary, &target).map_err(io_error)
    }

    pub fn index_path(&mut self, path: impl AsRef<Path>) -> Result<IndexReport, ShadowError> {
        let path = path.as_ref();
        let mut report = IndexReport::default();
        let candidates: Vec<PathBuf> = if path.is_file() {
            vec![path.to_path_buf()]
        } else {
            WalkDir::new(path)
                .follow_links(false)
                .into_iter()
                .filter_map(Result::ok)
                .filter(|e| e.file_type().is_file())
                .map(|e| e.into_path())
                .collect()
        };
        for candidate in candidates {
            if !supported(&candidate) {
                report.skipped += 1;
                continue;
            }
            let metadata = match candidate.metadata() {
                Ok(value) if value.len() <= MAX_FILE_BYTES => value,
                _ => {
                    report.skipped += 1;
                    continue;
                }
            };
            let body = match fs::read_to_string(&candidate) {
                Ok(value) => value,
                Err(_) => {
                    report.failed += 1;
                    continue;
                }
            };
            let absolute = candidate.canonicalize().unwrap_or(candidate.clone());
            let title = candidate
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Untitled");
            let document = ShadowDocument {
                doc_id: absolute.to_string_lossy().to_string(),
                title: title.to_string(),
                body,
                domain: "note".into(),
                origin_vault_key: absolute.parent().map(|p| p.to_string_lossy().to_string()),
            };
            let _ = metadata;
            match self.upsert(document) {
                Ok(()) => report.indexed += 1,
                Err(_) => report.failed += 1,
            }
        }
        self.flush()?;
        Ok(report)
    }

    pub fn search_sidebar(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RecallResult>, ShadowError> {
        self.search(query, limit, RecallMode::Sidebar, None)
    }

    pub fn recall_ambient(
        &self,
        origin_note_id: Option<&str>,
        current_text: &str,
        limit: usize,
    ) -> Result<Vec<RecallResult>, ShadowError> {
        let text = current_text.trim();
        if text.chars().count() < 12 {
            return Ok(Vec::new());
        }
        let query = tail_chars(text, 600);
        self.search(&query, limit, RecallMode::Ambient, origin_note_id)
    }

    pub fn document_count(&self) -> usize {
        self.docs.len()
    }

    fn search(
        &self,
        query: &str,
        limit: usize,
        mode: RecallMode,
        excluded: Option<&str>,
    ) -> Result<Vec<RecallResult>, ShadowError> {
        if query.trim().is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let started = Instant::now();
        let pool = limit.saturating_mul(6).max(30);
        let lexical = self.lexical.search_notes(query, pool)?;
        let mut title: Vec<(String, f32)> = self
            .docs
            .values()
            .filter_map(|doc| {
                title_score(&doc.title, query).map(|score| (doc.doc_id.clone(), score))
            })
            .collect();
        title.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        title.truncate(pool);
        let query_vector = vectorize(query);
        let mut vector: Vec<(String, f32)> = self
            .docs
            .values()
            .map(|doc| {
                let value = format!("{}\n{}", doc.title, doc.body);
                (
                    doc.doc_id.clone(),
                    cosine(&query_vector, &vectorize(&value)),
                )
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();
        vector.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        vector.truncate(pool);

        let lexical_rank: Vec<String> = lexical.iter().map(|h| h.doc_id.clone()).collect();
        let title_rank: Vec<String> = title.iter().map(|h| h.0.clone()).collect();
        let vector_rank: Vec<String> = vector.iter().map(|h| h.0.clone()).collect();
        let mut scores: HashMap<String, f32> = HashMap::new();
        let mut channels: HashMap<String, HashSet<String>> = HashMap::new();
        for (name, weight, ranking) in [
            ("title", 1.35_f32, &title_rank),
            ("bm25", 1.0_f32, &lexical_rank),
            ("trigram-vector", 0.8_f32, &vector_rank),
        ] {
            for (rank, id) in ranking.iter().enumerate() {
                *scores.entry(id.clone()).or_default() += weight / (RRF_K + rank as f32 + 1.0);
                channels.entry(id.clone()).or_default().insert(name.into());
            }
        }
        let lexical_snippets: HashMap<&str, &str> = lexical
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.snippet.as_str()))
            .collect();
        let mut ranked: Vec<(String, f32)> = scores
            .into_iter()
            .filter(|(id, _)| excluded != Some(id.as_str()))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(limit);
        let elapsed_us = started.elapsed().as_micros();
        Ok(ranked
            .into_iter()
            .filter_map(|(id, score)| {
                let doc = self.docs.get(&id)?;
                let mut signal_names: Vec<String> = channels.remove(&id)?.into_iter().collect();
                signal_names.sort();
                Some(RecallResult {
                    doc_id: id,
                    title: doc.title.clone(),
                    snippet: lexical_snippets
                        .get(doc.doc_id.as_str())
                        .map(|s| (*s).to_string())
                        .unwrap_or_else(|| snippet(&doc.body, query)),
                    score,
                    channels: signal_names,
                    mode,
                    elapsed_us,
                })
            })
            .collect())
    }
}

fn load_catalog(path: &Path) -> Result<HashMap<String, ShadowDocument>, ShadowError> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    serde_json::from_slice(&fs::read(path).map_err(io_error)?).map_err(backend_error)
}

fn supported(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some(
            "md" | "markdown"
                | "txt"
                | "html"
                | "htm"
                | "rst"
                | "org"
                | "tex"
                | "rs"
                | "swift"
                | "py"
                | "js"
                | "ts"
                | "tsx"
                | "jsx"
                | "java"
                | "c"
                | "h"
                | "cpp"
                | "cs"
                | "go"
                | "rb"
                | "php"
                | "toml"
                | "yaml"
                | "yml"
                | "json"
                | "xml"
                | "css"
                | "sql"
                | "sh"
                | "ps1"
        )
    )
}

fn title_score(title: &str, query: &str) -> Option<f32> {
    let title = title.to_lowercase();
    let query = query.trim().to_lowercase();
    if title == query {
        Some(3.0)
    } else if title.starts_with(&query) {
        Some(2.0)
    } else if title.contains(&query) {
        Some(1.0)
    } else {
        None
    }
}

fn vectorize(text: &str) -> Vec<f32> {
    let normalized: Vec<char> = text
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect();
    let mut out = vec![0.0; VECTOR_DIM];
    if normalized.len() < 3 {
        return out;
    }
    for window in normalized.windows(3) {
        let mut hash: u64 = 1469598103934665603;
        for ch in window {
            hash = (hash ^ (*ch as u64)).wrapping_mul(1099511628211);
        }
        let index = hash as usize % VECTOR_DIM;
        out[index] += if hash & 1 == 0 { 1.0 } else { -1.0 };
    }
    let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut out {
            *value /= norm;
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>().max(0.0)
}

fn tail_chars(text: &str, maximum: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    chars[chars.len().saturating_sub(maximum)..]
        .iter()
        .collect()
}

fn snippet(body: &str, query: &str) -> String {
    const MAX: usize = 180;
    let lower = body.to_lowercase();
    let query = query.to_lowercase();
    let byte_center = lower.find(query.trim()).unwrap_or(0);
    let char_center = body[..byte_center.min(body.len())].chars().count();
    let chars: Vec<char> = body.chars().collect();
    let start = char_center.saturating_sub(MAX / 3);
    chars[start..(start + MAX).min(chars.len())]
        .iter()
        .collect()
}

fn io_error(error: std::io::Error) -> ShadowError {
    ShadowError::Io {
        detail: error.to_string(),
    }
}
fn backend_error(error: impl std::fmt::Display) -> ShadowError {
    ShadowError::Backend {
        detail: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_are_distinct_and_ambient_excludes_origin() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut engine = RecallEngine::open(dir.path()).unwrap();
        for (id, title, body) in [
            (
                "a",
                "Meteor notes",
                "The observatory tracks a meteor shower before dawn.",
            ),
            ("b", "Grocery list", "Apples, oats, and tea."),
        ] {
            engine
                .upsert(ShadowDocument {
                    doc_id: id.into(),
                    title: title.into(),
                    body: body.into(),
                    domain: "note".into(),
                    origin_vault_key: None,
                })
                .unwrap();
        }
        engine.flush().unwrap();
        assert_eq!(
            engine.search_sidebar("meteor", 3).unwrap()[0].mode,
            RecallMode::Sidebar
        );
        let ambient = engine
            .recall_ambient(
                Some("a"),
                "I am writing about the observatory and meteor shower",
                3,
            )
            .unwrap();
        assert!(ambient.iter().all(|hit| hit.doc_id != "a"));
        assert!(ambient.iter().all(|hit| hit.mode == RecallMode::Ambient));
    }

    #[test]
    fn short_ambient_text_does_not_interrupt_typing() {
        let dir = tempfile::TempDir::new().unwrap();
        let engine = RecallEngine::open(dir.path()).unwrap();
        assert!(
            engine
                .recall_ambient(None, "too short", 5)
                .unwrap()
                .is_empty()
        );
    }
}
