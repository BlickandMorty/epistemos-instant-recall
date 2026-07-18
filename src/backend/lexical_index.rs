//! W8.4.d — tantivy BM25 wrapper.
//!
//! Schema mirrors agent_core/src/storage/vault.rs:133-137 (the
//! production tantivy usage):
//!
//!   doc_id  STRING | STORED   exact-match doc identifier
//!   domain  STRING | STORED   "note" / "chat" filter
//!   title   TEXT   | STORED   full-text indexed + retrievable
//!   body    TEXT   | STORED   full-text indexed + retrievable
//!
//! V1 uses `RamDirectory` (in-memory only); W8.4.f wires
//! `MmapDirectory` persistence at
//! `<vault>/.epistemos/shadow/tantivy/`.
//!
//! Reader uses `ReloadPolicy::OnCommitWithDelay` (mirrors vault.rs:144-147).
//! Each insert/remove commits immediately for V1; W8.4.f batches.
//!
//! Defensive query parsing: tantivy's QueryParser errors on
//! operator-only / unicode-corner inputs (`":"`, `"AND"`, `"!@#"`).
//! The wrapper catches `QueryParserError` and returns Ok(empty) so
//! the Swift caller never sees `-1 InvalidInput` for what looks like
//! normal typing.

use std::path::Path;
use std::sync::Mutex;

use tantivy::{
    Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term,
    collector::TopDocs,
    directory::{MmapDirectory, RamDirectory},
    doc,
    query::{BooleanQuery, Occur, Query, QueryParser, TermQuery},
    schema::{Field, IndexRecordOption, STORED, STRING, Schema, TEXT, Value},
};

use crate::ShadowDocument;
use crate::error::ShadowError;

/// Tantivy writer heap budget — 15 MB is tantivy's documented minimum
/// (`writer(heap)` returns an error below this). Halo's V1 corpus
/// (~1M docs cap) writes infrequently; the larger 50 MB heap that
/// vault.rs historically used was carried forward without measurement.
/// Lowering to the floor cuts ~35 MB resident on idle without
/// observed write-throughput change in the W8 test suite.
const WRITER_HEAP_BYTES: usize = 15_000_000;

pub struct LexicalIndex {
    index: Index,
    reader: IndexReader,
    writer: Mutex<IndexWriter>,
    field_doc_id: Field,
    field_domain: Field,
    field_title: Field,
    field_body: Field,
}

/// One lexical search hit. Score is tantivy's BM25 raw score.
#[derive(Debug, Clone)]
pub struct LexicalHit {
    pub doc_id: String,
    pub title: String,
    pub body: String,
    pub score: f32,
}

impl LexicalIndex {
    /// Build a fresh in-memory tantivy index. Useful for tests +
    /// short-lived builds. Production uses `open_at(&path)` to back
    /// the index by an MmapDirectory under
    /// `<vault>/.epistemos/shadow/tantivy/`.
    pub fn new() -> Result<Self, ShadowError> {
        Self::build_from_directory(Box::new(RamDirectory::create()))
    }

    /// Open (or create) an MmapDirectory-backed tantivy index at the
    /// given path. The directory is created if it doesn't exist.
    /// W8.4.f canonical persistence path.
    pub fn open_at(path: &Path) -> Result<Self, ShadowError> {
        std::fs::create_dir_all(path).map_err(|e| ShadowError::Io {
            detail: format!("create_dir_all({path:?}) failed: {e}"),
        })?;
        let directory = MmapDirectory::open(path).map_err(|e| ShadowError::Io {
            detail: format!("MmapDirectory::open({path:?}) failed: {e}"),
        })?;
        Self::build_from_directory(Box::new(directory))
    }

    /// Walk every stored document and emit `(doc_id, domain)` pairs.
    /// Used at startup to rebuild the doc_id ↔ row_key map for the
    /// VectorIndex (usearch persists vectors but not the mapping).
    /// Caps at 1M docs per Halo's V1 vault budget — adjust if a vault
    /// genuinely exceeds that.
    pub fn iter_doc_ids(&self) -> Result<Vec<(String, String)>, ShadowError> {
        use tantivy::query::AllQuery;
        const MAX_DOCS: usize = 1_000_000;
        let searcher = self.reader.searcher();
        let total = searcher.num_docs() as usize;
        if total == 0 {
            return Ok(Vec::new());
        }
        let limit = total.min(MAX_DOCS);
        let top = searcher
            .search(&AllQuery, &TopDocs::with_limit(limit))
            .map_err(|e| ShadowError::Backend {
                detail: format!("tantivy iter_doc_ids search failed: {e}"),
            })?;
        let mut out = Vec::with_capacity(top.len());
        for (_score, address) in top {
            let document: TantivyDocument =
                searcher.doc(address).map_err(|e| ShadowError::Backend {
                    detail: format!("tantivy doc fetch failed: {e}"),
                })?;
            let doc_id = stored_text(&document, self.field_doc_id);
            let domain = stored_text(&document, self.field_domain);
            if !doc_id.is_empty() {
                out.push((doc_id, domain));
            }
        }
        Ok(out)
    }

    fn build_from_directory(directory: Box<dyn tantivy::Directory>) -> Result<Self, ShadowError> {
        let mut schema_builder = Schema::builder();
        let field_doc_id = schema_builder.add_text_field("doc_id", STRING | STORED);
        let field_domain = schema_builder.add_text_field("domain", STRING | STORED);
        let field_title = schema_builder.add_text_field("title", TEXT | STORED);
        let field_body = schema_builder.add_text_field("body", TEXT | STORED);
        let schema = schema_builder.build();

        let index = Index::open_or_create(directory, schema).map_err(|e| ShadowError::Backend {
            detail: format!("tantivy::Index::open_or_create failed: {e}"),
        })?;
        // Manual reload so writes become visible synchronously after
        // each commit (insert / remove call `reader.reload()` explicitly
        // before returning). The OnCommitWithDelay default is async +
        // bites tests that read immediately after a write; manual
        // matches the V1 single-threaded write pattern exactly.
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| ShadowError::Backend {
                detail: format!("tantivy reader builder failed: {e}"),
            })?;
        let writer = index
            .writer(WRITER_HEAP_BYTES)
            .map_err(|e| ShadowError::Backend {
                detail: format!("tantivy writer init failed: {e}"),
            })?;

        Ok(Self {
            index,
            reader,
            writer: Mutex::new(writer),
            field_doc_id,
            field_domain,
            field_title,
            field_body,
        })
    }

    /// Insert (or replace) a document. Mirrors vault.rs:200's
    /// delete-then-add pattern so duplicate doc_ids don't pile up.
    pub fn insert(&self, doc: &ShadowDocument) -> Result<(), ShadowError> {
        let mut writer = self.writer.lock().map_err(|_| ShadowError::Backend {
            detail: "tantivy writer lock poisoned".into(),
        })?;
        writer.delete_term(Term::from_field_text(self.field_doc_id, &doc.doc_id));
        writer
            .add_document(doc!(
                self.field_doc_id => doc.doc_id.clone(),
                self.field_domain => doc.domain.clone(),
                self.field_title  => doc.title.clone(),
                self.field_body   => doc.body.clone(),
            ))
            .map_err(|e| ShadowError::Backend {
                detail: format!("tantivy add_document failed: {e}"),
            })?;
        writer.commit().map_err(|e| ShadowError::Backend {
            detail: format!("tantivy commit failed: {e}"),
        })?;
        // Manual-reload-policy reader: refresh the searcher so
        // subsequent searches see the new document.
        self.reader.reload().map_err(|e| ShadowError::Backend {
            detail: format!("tantivy reader reload failed: {e}"),
        })?;
        Ok(())
    }

    /// Remove every document with `doc_id`. Idempotent — removing an
    /// unknown id returns Ok.
    pub fn remove(&self, doc_id: &str) -> Result<(), ShadowError> {
        let mut writer = self.writer.lock().map_err(|_| ShadowError::Backend {
            detail: "tantivy writer lock poisoned".into(),
        })?;
        writer.delete_term(Term::from_field_text(self.field_doc_id, doc_id));
        writer.commit().map_err(|e| ShadowError::Backend {
            detail: format!("tantivy commit failed: {e}"),
        })?;
        self.reader.reload().map_err(|e| ShadowError::Backend {
            detail: format!("tantivy reader reload failed: {e}"),
        })?;
        Ok(())
    }

    /// BM25 search filtered by domain. Defensive against operator-only
    /// or unicode-corner inputs — those return Ok(empty) instead of
    /// surfacing a -1 InvalidInput to the Swift host.
    pub fn search(
        &self,
        query: &str,
        domain: &str,
        limit: usize,
    ) -> Result<Vec<LexicalHit>, ShadowError> {
        if query.trim().is_empty() || limit == 0 {
            return Ok(Vec::new());
        }

        let searcher = self.reader.searcher();
        let parser = QueryParser::for_index(&self.index, vec![self.field_title, self.field_body]);

        // Try the parsed query; on QueryParserError, fall back to a
        // bag-of-tokens TermQuery on the body field. Either way, the
        // fallback gets AND-ed with the domain filter.
        let body_query: Box<dyn Query> = match parser.parse_query(query) {
            Ok(q) => q,
            Err(_) => fallback_term_query(query, self.field_body),
        };
        let domain_query = Box::new(TermQuery::new(
            Term::from_field_text(self.field_domain, domain),
            IndexRecordOption::Basic,
        ));

        let combined =
            BooleanQuery::new(vec![(Occur::Must, body_query), (Occur::Must, domain_query)]);

        let top_docs = searcher
            .search(&combined, &TopDocs::with_limit(limit))
            .map_err(|e| ShadowError::Backend {
                detail: format!("tantivy search failed: {e}"),
            })?;

        let mut hits: Vec<LexicalHit> = Vec::with_capacity(top_docs.len());
        for (score, address) in top_docs {
            let document: TantivyDocument =
                searcher.doc(address).map_err(|e| ShadowError::Backend {
                    detail: format!("tantivy doc fetch failed: {e}"),
                })?;
            let doc_id = stored_text(&document, self.field_doc_id);
            let title = stored_text(&document, self.field_title);
            let body = stored_text(&document, self.field_body);
            hits.push(LexicalHit {
                doc_id,
                title,
                body,
                score,
            });
        }
        Ok(hits)
    }

    /// Doc count via the reader. Useful for stats; the reader is
    /// updated on each commit so this is current.
    pub fn doc_count(&self) -> u64 {
        self.reader.searcher().num_docs()
    }
}

/// Build a permissive fallback query that ANDs together every
/// alphanumeric token from the user's input. Used when QueryParser
/// can't parse the input as Lucene-style syntax (operator-only,
/// punctuation-only, unicode-corner) — the user's intent in those
/// cases is "find these characters somewhere" not "syntax error."
fn fallback_term_query(query: &str, body_field: Field) -> Box<dyn Query> {
    let tokens: Vec<String> = query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    if tokens.is_empty() {
        // Pure-punctuation input → match nothing rather than
        // panicking the searcher with an empty BooleanQuery.
        return Box::new(BooleanQuery::new(Vec::new()));
    }
    let clauses: Vec<(Occur, Box<dyn Query>)> = tokens
        .into_iter()
        .map(|token| {
            let q: Box<dyn Query> = Box::new(TermQuery::new(
                Term::from_field_text(body_field, &token),
                IndexRecordOption::Basic,
            ));
            (Occur::Must, q)
        })
        .collect();
    Box::new(BooleanQuery::new(clauses))
}

fn stored_text(document: &TantivyDocument, field: Field) -> String {
    document
        .get_first(field)
        .and_then(|value| value.as_str().map(|s| s.to_string()))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(id: &str, title: &str, body: &str) -> ShadowDocument {
        ShadowDocument {
            doc_id: id.to_string(),
            domain: "note".to_string(),
            title: title.to_string(),
            body: body.to_string(),
            origin_vault_key: None,
        }
    }

    #[test]
    fn insert_then_search_returns_hit() {
        let idx = LexicalIndex::new().unwrap();
        idx.insert(&note("a", "Quarterly report", "revenue grew by 12 percent"))
            .unwrap();
        let hits = idx.search("revenue", "note", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "a");
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn search_filters_by_domain() {
        let idx = LexicalIndex::new().unwrap();
        idx.insert(&note("a", "x", "report")).unwrap();
        idx.insert(&ShadowDocument {
            doc_id: "b".to_string(),
            domain: "chat".to_string(),
            title: "y".to_string(),
            body: "report".to_string(),
            origin_vault_key: None,
        })
        .unwrap();
        let note_hits = idx.search("report", "note", 10).unwrap();
        let chat_hits = idx.search("report", "chat", 10).unwrap();
        assert_eq!(note_hits.len(), 1);
        assert_eq!(note_hits[0].doc_id, "a");
        assert_eq!(chat_hits.len(), 1);
        assert_eq!(chat_hits[0].doc_id, "b");
    }

    #[test]
    fn remove_excludes_from_search() {
        let idx = LexicalIndex::new().unwrap();
        idx.insert(&note("a", "x", "important")).unwrap();
        idx.remove("a").unwrap();
        let hits = idx.search("important", "note", 10).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn remove_unknown_is_idempotent() {
        let idx = LexicalIndex::new().unwrap();
        idx.remove("never-inserted").unwrap();
        idx.remove("never-inserted").unwrap();
        assert_eq!(idx.doc_count(), 0);
    }

    #[test]
    fn empty_query_returns_empty() {
        let idx = LexicalIndex::new().unwrap();
        idx.insert(&note("a", "x", "body")).unwrap();
        assert!(idx.search("", "note", 10).unwrap().is_empty());
        assert!(idx.search("   ", "note", 10).unwrap().is_empty());
    }

    #[test]
    fn limit_zero_returns_empty() {
        let idx = LexicalIndex::new().unwrap();
        idx.insert(&note("a", "x", "body")).unwrap();
        assert!(idx.search("body", "note", 0).unwrap().is_empty());
    }

    #[test]
    fn special_chars_in_query_dont_panic() {
        let idx = LexicalIndex::new().unwrap();
        idx.insert(&note("a", "x", "body")).unwrap();
        // QueryParser hostile inputs — must return Ok(empty), NOT bubble Err
        for query in [":", "!@#$", "AND", "OR", "()", "[]", "+-*/"] {
            let result = idx.search(query, "note", 10);
            assert!(
                result.is_ok(),
                "punctuation query '{query}' MUST NOT bubble parser error; got {result:?}"
            );
        }
    }

    #[test]
    fn insert_replaces_document_for_same_doc_id() {
        let idx = LexicalIndex::new().unwrap();
        idx.insert(&note("a", "first", "first body")).unwrap();
        idx.insert(&note("a", "second", "second body")).unwrap();
        assert_eq!(idx.doc_count(), 1, "insert MUST replace, not duplicate");
        let hits = idx.search("second", "note", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].title, "second");
    }
}
