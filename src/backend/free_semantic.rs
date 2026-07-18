//! Candidate-neutral contracts for Free's paragraph semantic retrieval.
//!
//! This module deliberately has no model, network, FFI, ANN, or filesystem
//! loader. It defines the note-only data and ranking contracts that every
//! eventual local embedding candidate must obey. It is not production semantic
//! search until the evidence-gated candidate selection and integration land.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{self, Write as _},
};

use sha2::{Digest, Sha256};
use thiserror::Error;

pub const CHUNK_FORMAT_VERSION: u32 = 2;
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const DIGEST_ALGORITHM: &str = "sha256-v1";

const MAX_NOTE_BYTES: usize = 1_048_576;
const MAX_CHUNK_BYTES: usize = 2_048;
const MAX_CHUNKS_PER_NOTE: usize = 128;
const MAX_OVERLAP_BYTES: usize = 256;
const MAX_VECTOR_DIMENSION: usize = 4_096;
const MAX_VECTOR_COUNT: usize = 16_384;
const MAX_ID_BYTES: usize = 256;
const MAX_QUERY_LIMIT: usize = 64;
const MAX_QUERY_BYTES: usize = 16 * 1024;
const MAX_CHANNEL_HITS: usize = 256;
const MAX_EXACT_TITLE_BYTES: usize = 2_048;
const SEARCH_REQUEST_FORMAT_VERSION: u32 = 2;
const LEXICAL_INPUT_POLICY: &str = "free-lexical-escaped-query-v1";
const SEMANTIC_INPUT_POLICY: &str = "free-semantic-exact-utf8-v1";
const LEXICAL_INPUT_POLICY_VERSION: u32 = 1;
const SEMANTIC_INPUT_POLICY_VERSION: u32 = 1;
const FLOAT_EPSILON: f64 = 1.0e-12;

/// Strictly note-only input. Legacy wire discriminators deliberately have no
/// constructor here, so historical chat data cannot enter chunking.
#[derive(Clone, PartialEq, Eq)]
pub struct NoteInput {
    pub vault_id: String,
    pub note_id: String,
    pub title: String,
    pub body: String,
}

impl fmt::Debug for NoteInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NoteInput")
            .field("title_bytes", &self.title.len())
            .field("body_bytes", &self.body.len())
            .finish()
    }
}

impl NoteInput {
    pub fn new(
        vault_id: impl Into<String>,
        note_id: impl Into<String>,
        title: impl Into<String>,
        body: impl Into<String>,
    ) -> Result<Self, FreeSemanticError> {
        let vault_id = canonical_identity(vault_id.into(), IdentityKind::Vault)?;
        let note_id = canonical_identity(note_id.into(), IdentityKind::Note)?;
        let title = title.into();
        let body = body.into();
        if title
            .len()
            .checked_add(body.len())
            .map_or(true, |byte_len| byte_len > MAX_NOTE_BYTES)
        {
            return Err(FreeSemanticError::NoteTooLarge {
                limit: MAX_NOTE_BYTES,
            });
        }
        Ok(Self {
            vault_id,
            note_id,
            title,
            body,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkingPolicy {
    pub max_chunk_bytes: usize,
    pub max_chunks_per_note: usize,
    pub overlap_bytes: usize,
}

impl Default for ChunkingPolicy {
    fn default() -> Self {
        Self {
            max_chunk_bytes: MAX_CHUNK_BYTES,
            max_chunks_per_note: MAX_CHUNKS_PER_NOTE,
            overlap_bytes: 128,
        }
    }
}

impl ChunkingPolicy {
    fn validate(&self) -> Result<(), FreeSemanticError> {
        if self.max_chunk_bytes == 0 || self.max_chunk_bytes > MAX_CHUNK_BYTES {
            return Err(FreeSemanticError::InvalidChunkPolicy);
        }
        if self.max_chunks_per_note == 0 || self.max_chunks_per_note > MAX_CHUNKS_PER_NOTE {
            return Err(FreeSemanticError::InvalidChunkPolicy);
        }
        if self.overlap_bytes >= self.max_chunk_bytes || self.overlap_bytes > MAX_OVERLAP_BYTES {
            return Err(FreeSemanticError::InvalidChunkPolicy);
        }
        Ok(())
    }

    fn digest(&self) -> String {
        sha256_digest(
            "chunking-policy-v1",
            &[
                &self.max_chunk_bytes.to_string(),
                &self.max_chunks_per_note.to_string(),
                &self.overlap_bytes.to_string(),
            ],
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ChunkKind {
    Title,
    Heading,
    Paragraph,
    Code,
}

impl ChunkKind {
    fn tag(self) -> &'static str {
        match self {
            Self::Title => "title",
            Self::Heading => "heading",
            Self::Paragraph => "paragraph",
            Self::Code => "code",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BodyByteRange {
    pub start_byte: usize,
    pub end_byte: usize,
}

#[derive(Clone, PartialEq, Eq)]
pub struct ParagraphChunk {
    /// Stable only while the logical occurrence is unchanged. It is distinct
    /// from the page identity and the full note revision digest.
    pub logical_id: String,
    pub chunk_id: String,
    pub vault_id: String,
    pub note_id: String,
    pub title: String,
    pub text: String,
    pub kind: ChunkKind,
    /// Title chunks have no body range; every body chunk has an exact trimmed
    /// UTF-8 byte range into `NoteInput.body`.
    pub body_range: Option<BodyByteRange>,
    pub content_digest: String,
    pub note_revision_digest: String,
    /// Occurrence is reusable only for unique content. Ambiguous duplicate
    /// chunks are revision-scoped before an old vector can be reused.
    pub occurrence: usize,
    pub format_version: u32,
}

impl fmt::Debug for ParagraphChunk {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ParagraphChunk")
            .field("kind", &self.kind)
            .field("text_bytes", &self.text.len())
            .field("has_body_range", &self.body_range.is_some())
            .field("occurrence", &self.occurrence)
            .field("format_version", &self.format_version)
            .finish()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectionPartialReason {
    ChunkLimit,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectionStatus {
    Complete,
    Partial {
        omitted_from_body_byte: usize,
        reason: ProjectionPartialReason,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkCoverage {
    pub total_body_bytes: usize,
    pub indexed_body_bytes: usize,
    pub status: ProjectionStatus,
}

#[derive(Clone, PartialEq, Eq)]
pub struct ChunkProjection {
    pub vault_id: String,
    pub note_id: String,
    /// Bounded canonical source retained only while this in-memory projection
    /// is validated and staged; it is not a derived-index persistence format.
    pub canonical_title: String,
    pub canonical_body: String,
    pub note_revision_digest: String,
    /// The bounded canonical policy used to construct this exact projection.
    /// Its digest is retained separately for the generation receipt.
    pub chunking_policy: ChunkingPolicy,
    pub chunking_policy_digest: String,
    pub body_byte_len: usize,
    pub chunks: Vec<ParagraphChunk>,
    pub coverage: ChunkCoverage,
}

impl fmt::Debug for ChunkProjection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChunkProjection")
            .field("title_bytes", &self.canonical_title.len())
            .field("body_bytes", &self.canonical_body.len())
            .field("chunk_count", &self.chunks.len())
            .field("coverage", &self.coverage)
            .finish()
    }
}

/// Project a note into deterministic title, heading, paragraph, and fenced
/// code chunks. A note that exceeds its chosen chunk budget returns a truthful
/// partial projection rather than pretending the whole body was indexed.
pub fn chunk_note(
    note: &NoteInput,
    policy: &ChunkingPolicy,
) -> Result<ChunkProjection, FreeSemanticError> {
    policy.validate()?;
    if note
        .title
        .len()
        .checked_add(note.body.len())
        .map_or(true, |byte_len| byte_len > MAX_NOTE_BYTES)
    {
        return Err(FreeSemanticError::NoteTooLarge {
            limit: MAX_NOTE_BYTES,
        });
    }

    let note_revision_digest = sha256_digest(
        "note-revision-v1",
        &[&note.vault_id, &note.note_id, &note.title, &note.body],
    );
    let mut chunks = Vec::new();
    let mut occurrences: BTreeMap<(ChunkKind, String), usize> = BTreeMap::new();
    let mut omitted_from = None;

    if !note.title.trim().is_empty()
        && !append_chunk(
            &mut chunks,
            &mut occurrences,
            note,
            ChunkKind::Title,
            &note.title,
            None,
            &note_revision_digest,
            policy,
        )?
    {
        omitted_from = Some(0);
    }

    if omitted_from.is_none() {
        'blocks: for block in markdown_blocks(&note.body) {
            for segment in split_bounded(
                &note.body[block.start_byte..block.end_byte],
                policy.max_chunk_bytes,
                policy.overlap_bytes,
            ) {
                let start_byte = block.start_byte + segment.start_byte;
                let end_byte = block.start_byte + segment.end_byte;
                let (text, range) = trimmed_body_text(&note.body, start_byte, end_byte)?;
                if text.is_empty() {
                    continue;
                }
                if !append_chunk(
                    &mut chunks,
                    &mut occurrences,
                    note,
                    block.kind,
                    text,
                    Some(range),
                    &note_revision_digest,
                    policy,
                )? {
                    omitted_from = Some(first_unrepresented_body_byte(&chunks, start_byte));
                    break 'blocks;
                }
            }
        }
    }

    let indexed_body_bytes = covered_body_bytes(&chunks);
    finalize_ambiguous_chunk_identities(&mut chunks, &note_revision_digest);
    let status = match omitted_from {
        Some(omitted_from_body_byte) => ProjectionStatus::Partial {
            omitted_from_body_byte,
            reason: ProjectionPartialReason::ChunkLimit,
        },
        None => ProjectionStatus::Complete,
    };
    Ok(ChunkProjection {
        vault_id: note.vault_id.clone(),
        note_id: note.note_id.clone(),
        canonical_title: note.title.clone(),
        canonical_body: note.body.clone(),
        note_revision_digest,
        chunking_policy: policy.clone(),
        chunking_policy_digest: policy.digest(),
        body_byte_len: note.body.len(),
        chunks,
        coverage: ChunkCoverage {
            total_body_bytes: note.body.len(),
            indexed_body_bytes,
            status,
        },
    })
}

fn append_chunk(
    chunks: &mut Vec<ParagraphChunk>,
    occurrences: &mut BTreeMap<(ChunkKind, String), usize>,
    note: &NoteInput,
    kind: ChunkKind,
    raw_text: &str,
    body_range: Option<BodyByteRange>,
    note_revision_digest: &str,
    policy: &ChunkingPolicy,
) -> Result<bool, FreeSemanticError> {
    let text = raw_text.trim();
    if text.is_empty() {
        return Ok(true);
    }
    if chunks.len() >= policy.max_chunks_per_note {
        return Ok(false);
    }
    if text.len() > policy.max_chunk_bytes {
        return Err(FreeSemanticError::ChunkExceedsPolicy {
            limit: policy.max_chunk_bytes,
        });
    }
    let content_digest = sha256_digest("chunk-content-v1", &[text]);
    let occurrence_key = (kind, content_digest.clone());
    let occurrence = match occurrences.get_mut(&occurrence_key) {
        Some(value) => {
            *value = value
                .checked_add(1)
                .ok_or(FreeSemanticError::ChunkOccurrenceOverflow)?;
            *value
        }
        None => {
            occurrences.insert(occurrence_key, 0);
            0
        }
    };
    let logical_id = logical_chunk_id(
        &note.vault_id,
        &note.note_id,
        kind,
        &content_digest,
        occurrence,
        None,
    );
    let chunk_id = format!("pc{CHUNK_FORMAT_VERSION}:{logical_id}");
    chunks.push(ParagraphChunk {
        logical_id,
        chunk_id,
        vault_id: note.vault_id.clone(),
        note_id: note.note_id.clone(),
        title: note.title.clone(),
        text: text.to_string(),
        kind,
        body_range,
        content_digest,
        note_revision_digest: note_revision_digest.to_string(),
        occurrence,
        format_version: CHUNK_FORMAT_VERSION,
    });
    Ok(true)
}

fn logical_chunk_id(
    vault_id: &str,
    note_id: &str,
    kind: ChunkKind,
    content_digest: &str,
    occurrence: usize,
    ambiguity_revision: Option<&str>,
) -> String {
    let occurrence_text = occurrence.to_string();
    match ambiguity_revision {
        Some(revision) => sha256_digest(
            "ambiguous-logical-chunk-v2",
            &[
                vault_id,
                note_id,
                kind.tag(),
                content_digest,
                &occurrence_text,
                revision,
            ],
        ),
        None => sha256_digest(
            "logical-chunk-v2",
            &[
                vault_id,
                note_id,
                kind.tag(),
                content_digest,
                &occurrence_text,
            ],
        ),
    }
}

fn finalize_ambiguous_chunk_identities(chunks: &mut [ParagraphChunk], note_revision_digest: &str) {
    let mut counts = BTreeMap::new();
    for chunk in chunks.iter() {
        *counts
            .entry((chunk.kind, chunk.content_digest.clone()))
            .or_insert(0usize) += 1;
    }
    for chunk in chunks {
        let ambiguous = counts
            .get(&(chunk.kind, chunk.content_digest.clone()))
            .is_some_and(|count| *count > 1);
        let logical_id = logical_chunk_id(
            &chunk.vault_id,
            &chunk.note_id,
            chunk.kind,
            &chunk.content_digest,
            chunk.occurrence,
            ambiguous.then_some(note_revision_digest),
        );
        chunk.chunk_id = format!("pc{CHUNK_FORMAT_VERSION}:{logical_id}");
        chunk.logical_id = logical_id;
    }
}

#[derive(Clone, Copy)]
struct MarkdownBlock {
    kind: ChunkKind,
    start_byte: usize,
    end_byte: usize,
}

fn markdown_blocks(body: &str) -> Vec<MarkdownBlock> {
    let mut blocks = Vec::new();
    let mut paragraph_start = None;
    let mut code_start = None;
    let mut in_code = false;
    let mut cursor = 0;

    for line in body.split_inclusive('\n') {
        let line_start = cursor;
        cursor += line.len();
        let line_end = cursor;
        let trimmed = line.trim();
        let fence = trimmed.starts_with("```") || trimmed.starts_with("~~~");

        if in_code {
            if fence {
                blocks.push(MarkdownBlock {
                    kind: ChunkKind::Code,
                    start_byte: code_start.take().unwrap_or(line_start),
                    end_byte: line_end,
                });
                in_code = false;
            }
            continue;
        }

        if fence {
            flush_paragraph(&mut blocks, &mut paragraph_start, line_start);
            code_start = Some(line_start);
            in_code = true;
        } else if trimmed.is_empty() {
            flush_paragraph(&mut blocks, &mut paragraph_start, line_start);
        } else if trimmed.starts_with('#') {
            flush_paragraph(&mut blocks, &mut paragraph_start, line_start);
            blocks.push(MarkdownBlock {
                kind: ChunkKind::Heading,
                start_byte: line_start,
                end_byte: line_end,
            });
        } else {
            paragraph_start.get_or_insert(line_start);
        }
    }

    if in_code {
        blocks.push(MarkdownBlock {
            kind: ChunkKind::Code,
            start_byte: code_start.unwrap_or(cursor),
            end_byte: body.len(),
        });
    } else {
        flush_paragraph(&mut blocks, &mut paragraph_start, body.len());
    }
    blocks
}

fn flush_paragraph(blocks: &mut Vec<MarkdownBlock>, start: &mut Option<usize>, end_byte: usize) {
    if let Some(start_byte) = start.take() {
        blocks.push(MarkdownBlock {
            kind: ChunkKind::Paragraph,
            start_byte,
            end_byte,
        });
    }
}

#[derive(Clone, Copy)]
struct LocalRange {
    start_byte: usize,
    end_byte: usize,
}

fn split_bounded(text: &str, max_bytes: usize, overlap_bytes: usize) -> Vec<LocalRange> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut segments = Vec::new();
    let mut start_byte = 0;
    while start_byte < text.len() {
        let hard_end = previous_char_boundary(text, (start_byte + max_bytes).min(text.len()));
        let end_byte = preferred_break(text, start_byte, hard_end).unwrap_or(hard_end);
        let end_byte = if end_byte > start_byte {
            end_byte
        } else {
            next_char_boundary(text, start_byte)
        };
        segments.push(LocalRange {
            start_byte,
            end_byte,
        });
        if end_byte == text.len() {
            break;
        }
        let overlapping_start =
            previous_char_boundary(text, end_byte.saturating_sub(overlap_bytes));
        start_byte = if overlapping_start > start_byte {
            overlapping_start
        } else {
            end_byte
        };
    }
    segments
}

fn preferred_break(text: &str, start_byte: usize, end_byte: usize) -> Option<usize> {
    let mut preferred = None;
    for (relative, character) in text[start_byte..end_byte].char_indices() {
        if character.is_whitespace() || matches!(character, '.' | ',' | ';' | ':' | '!' | '?') {
            preferred = Some(start_byte + relative + character.len_utf8());
        }
    }
    preferred.filter(|boundary| *boundary > start_byte)
}

fn previous_char_boundary(text: &str, mut byte: usize) -> usize {
    byte = byte.min(text.len());
    while byte > 0 && !text.is_char_boundary(byte) {
        byte -= 1;
    }
    byte
}

fn next_char_boundary(text: &str, start_byte: usize) -> usize {
    text.char_indices()
        .map(|(index, _)| index)
        .find(|index| *index > start_byte)
        .unwrap_or(text.len())
}

fn trimmed_body_text(
    body: &str,
    start_byte: usize,
    end_byte: usize,
) -> Result<(&str, BodyByteRange), FreeSemanticError> {
    if start_byte > end_byte
        || end_byte > body.len()
        || !body.is_char_boundary(start_byte)
        || !body.is_char_boundary(end_byte)
    {
        return Err(FreeSemanticError::InvalidBodyRange);
    }
    let raw = &body[start_byte..end_byte];
    let leading_trimmed = raw.trim_start();
    let leading = raw.len() - leading_trimmed.len();
    let trimmed = leading_trimmed.trim_end();
    let trimmed_end = start_byte + leading + trimmed.len();
    Ok((
        trimmed,
        BodyByteRange {
            start_byte: start_byte + leading,
            end_byte: trimmed_end,
        },
    ))
}

fn covered_body_bytes(chunks: &[ParagraphChunk]) -> usize {
    let mut ranges: Vec<_> = chunks
        .iter()
        .filter_map(|chunk| chunk.body_range.clone())
        .collect();
    ranges.sort_by_key(|range| (range.start_byte, range.end_byte));
    let mut total = 0;
    let mut current: Option<BodyByteRange> = None;
    for range in ranges {
        match current.as_mut() {
            Some(active) if range.start_byte <= active.end_byte => {
                active.end_byte = active.end_byte.max(range.end_byte);
            }
            Some(active) => {
                total += active.end_byte - active.start_byte;
                current = Some(range);
            }
            None => current = Some(range),
        }
    }
    if let Some(active) = current {
        total += active.end_byte - active.start_byte;
    }
    total
}

fn first_unrepresented_body_byte(chunks: &[ParagraphChunk], fallback: usize) -> usize {
    chunks
        .iter()
        .filter_map(|chunk| chunk.body_range.as_ref().map(|range| range.end_byte))
        .max()
        .unwrap_or(fallback)
        .max(fallback)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorNormalization {
    /// The injected fixture adapter normalizes each vector while computing
    /// cosine. This is not a claim about any selected model's output format.
    InjectedCosineL2,
}

impl VectorNormalization {
    fn tag(self) -> &'static str {
        match self {
            Self::InjectedCosineL2 => "injected-cosine-l2-v1",
        }
    }
}

/// A channel artifact receipt is supplied by the eventual lexical/vector
/// staging adapter. This contract binds that external digest to the exact
/// in-memory note/chunk state before publishing; it does not pretend to verify
/// a file, mmap, or ANN index that this module deliberately does not open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChannelReceipt {
    Pending {
        artifact_digest: String,
    },
    Bound {
        artifact_digest: String,
        staged_state_digest: String,
    },
}

impl ChannelReceipt {
    fn pending_artifact_digest(&self) -> Option<&str> {
        match self {
            Self::Pending { artifact_digest } => Some(artifact_digest),
            Self::Bound { .. } => None,
        }
    }

    fn bound_parts(&self) -> Option<(&str, &str)> {
        match self {
            Self::Pending { .. } => None,
            Self::Bound {
                artifact_digest,
                staged_state_digest,
            } => Some((artifact_digest, staged_state_digest)),
        }
    }
}

/// A lexical staging assertion supplied by a later persistence adapter. It is
/// deliberately narrower than a generation manifest: delete/reset may update
/// only the exact lexical artifact they staged, never a model, vector, rank,
/// or chunking contract. This pure module does not attest that bytes were
/// durably written; that remains the persistence adapter's responsibility.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LexicalStagingReceipt {
    artifact_digest: String,
}

impl LexicalStagingReceipt {
    pub fn new(artifact_digest: impl Into<String>) -> Result<Self, FreeSemanticError> {
        let receipt = Self {
            artifact_digest: artifact_digest.into(),
        };
        if !is_sha256_digest(&receipt.artifact_digest) {
            return Err(FreeSemanticError::InvalidManifest);
        }
        Ok(receipt)
    }

    fn pending_channel_receipt(&self) -> ChannelReceipt {
        ChannelReceipt::Pending {
            artifact_digest: self.artifact_digest.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationManifest {
    pub schema_version: u32,
    pub digest_algorithm: String,
    pub vault_id: String,
    pub generation: u64,
    /// The cohesive vector contract is absent until a candidate is selected
    /// and a vector generation is actually ready to publish.
    pub model_descriptor_digest: Option<String>,
    pub dimension: Option<usize>,
    pub chunk_format_version: u32,
    pub lexical_generation: u64,
    pub vector_generation: Option<u64>,
    pub rank_policy_version: u32,
    pub rank_policy_rrf_k: usize,
    pub rank_policy_digest: String,
    pub normalization: Option<VectorNormalization>,
    pub semantic_availability: SemanticAvailability,
    pub lexical_receipt: ChannelReceipt,
    pub vector_receipt: Option<ChannelReceipt>,
    /// Generated only when a staged projection is atomically published.
    pub chunking_policy_digest: Option<String>,
    /// Generated only when a staged catalog is atomically published.
    pub chunk_map_digest: Option<String>,
    /// Generated only when a staged catalog is atomically published.
    pub note_set_digest: Option<String>,
}

impl GenerationManifest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vault_id: impl Into<String>,
        generation: u64,
        model_descriptor_digest: Option<String>,
        dimension: Option<usize>,
        chunk_format_version: u32,
        lexical_generation: u64,
        vector_generation: Option<u64>,
        rank_policy: RankFusionPolicy,
        normalization: Option<VectorNormalization>,
        semantic_availability: SemanticAvailability,
        lexical_artifact_digest: impl Into<String>,
        vector_artifact_digest: Option<String>,
    ) -> Result<Self, FreeSemanticError> {
        let manifest = Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            digest_algorithm: DIGEST_ALGORITHM.to_string(),
            vault_id: canonical_identity(vault_id.into(), IdentityKind::Vault)?,
            generation,
            model_descriptor_digest,
            dimension,
            chunk_format_version,
            lexical_generation,
            vector_generation,
            rank_policy_version: rank_policy.version,
            rank_policy_rrf_k: rank_policy.rrf_k,
            rank_policy_digest: rank_policy.digest(),
            normalization,
            semantic_availability,
            lexical_receipt: ChannelReceipt::Pending {
                artifact_digest: lexical_artifact_digest.into(),
            },
            vector_receipt: vector_artifact_digest
                .map(|artifact_digest| ChannelReceipt::Pending { artifact_digest }),
            chunking_policy_digest: None,
            chunk_map_digest: None,
            note_set_digest: None,
        };
        manifest.validate_shape()?;
        Ok(manifest)
    }

    pub fn digest(&self) -> String {
        let (lexical_artifact_digest, lexical_state_digest) =
            receipt_digest_fields(&self.lexical_receipt);
        let (vector_artifact_digest, vector_state_digest) = self
            .vector_receipt
            .as_ref()
            .map(receipt_digest_fields)
            .unwrap_or(("none", "none"));
        sha256_digest(
            "semantic-generation-manifest-v1",
            &[
                &self.schema_version.to_string(),
                &self.digest_algorithm,
                &self.vault_id,
                &self.generation.to_string(),
                self.model_descriptor_digest.as_deref().unwrap_or("none"),
                &self
                    .dimension
                    .map(|dimension| dimension.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                &self.chunk_format_version.to_string(),
                &self.lexical_generation.to_string(),
                &self
                    .vector_generation
                    .map(|generation| generation.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                &self.rank_policy_version.to_string(),
                &self.rank_policy_rrf_k.to_string(),
                &self.rank_policy_digest,
                self.normalization
                    .map(VectorNormalization::tag)
                    .unwrap_or("none"),
                semantic_availability_tag(self.semantic_availability),
                lexical_artifact_digest,
                lexical_state_digest,
                vector_artifact_digest,
                vector_state_digest,
                self.chunking_policy_digest.as_deref().unwrap_or("none"),
                self.chunk_map_digest.as_deref().unwrap_or("none"),
                self.note_set_digest.as_deref().unwrap_or("none"),
            ],
        )
    }

    fn validate_shape(&self) -> Result<(), FreeSemanticError> {
        self.validate_base()?;
        if self.lexical_generation != self.generation
            || self.chunking_policy_digest.is_some()
            || self.chunk_map_digest.is_some()
            || self.note_set_digest.is_some()
            || !self
                .lexical_receipt
                .pending_artifact_digest()
                .is_some_and(is_sha256_digest)
            || !self.has_valid_pending_vector_receipt()
        {
            return Err(FreeSemanticError::InvalidManifest);
        }
        Ok(())
    }

    fn validate_base(&self) -> Result<(), FreeSemanticError> {
        if self.schema_version != MANIFEST_SCHEMA_VERSION
            || self.digest_algorithm != DIGEST_ALGORITHM
            || self.chunk_format_version != CHUNK_FORMAT_VERSION
            || self.generation == 0
            || self.rank_policy_version == 0
            || self.rank_policy_rrf_k == 0
            || !is_sha256_digest(&self.rank_policy_digest)
        {
            return Err(FreeSemanticError::InvalidManifest);
        }
        let rank_policy = RankFusionPolicy::new(self.rank_policy_version, self.rank_policy_rrf_k)
            .map_err(|_| FreeSemanticError::InvalidManifest)?;
        if rank_policy.digest() != self.rank_policy_digest {
            return Err(FreeSemanticError::InvalidManifest);
        }
        Ok(())
    }

    fn has_valid_pending_vector_receipt(&self) -> bool {
        match (
            self.semantic_availability.is_available(),
            self.model_descriptor_digest.as_deref(),
            self.dimension,
            self.normalization,
            self.vector_generation,
            self.vector_receipt.as_ref(),
        ) {
            (true, Some(descriptor), Some(dimension), Some(_), Some(generation), Some(receipt)) => {
                generation == self.generation
                    && dimension > 0
                    && dimension <= MAX_VECTOR_DIMENSION
                    && is_sha256_digest(descriptor)
                    && receipt
                        .pending_artifact_digest()
                        .is_some_and(is_sha256_digest)
            }
            (false, None, None, None, None, None) => true,
            _ => false,
        }
    }

    fn with_staged_state(
        mut self,
        chunking_policy_digest: String,
        chunks: &BTreeMap<String, ParagraphChunk>,
        chunks_by_note: &BTreeMap<String, BTreeSet<String>>,
    ) -> Result<Self, FreeSemanticError> {
        self.chunking_policy_digest = Some(chunking_policy_digest);
        self.chunk_map_digest = Some(chunk_map_digest(chunks));
        self.note_set_digest = Some(note_set_digest(chunks_by_note));
        let lexical_pending = self.lexical_receipt.clone();
        let vector_pending = self.vector_receipt.clone();
        let lexical_receipt = self.bind_staged_receipt(&lexical_pending, "lexical")?;
        let vector_receipt = vector_pending
            .as_ref()
            .map(|receipt| self.bind_staged_receipt(receipt, "vector"))
            .transpose()?;
        self.lexical_receipt = lexical_receipt;
        self.vector_receipt = vector_receipt;
        Ok(self)
    }

    fn validate_published_state(&self) -> Result<(), FreeSemanticError> {
        self.validate_base()?;
        if self.lexical_generation != self.generation
            || !self
                .chunking_policy_digest
                .as_deref()
                .is_some_and(is_sha256_digest)
            || !self
                .chunk_map_digest
                .as_deref()
                .is_some_and(is_sha256_digest)
            || !self
                .note_set_digest
                .as_deref()
                .is_some_and(is_sha256_digest)
            || !self.is_bound_to_staged_state(&self.lexical_receipt, "lexical")
            || !self.has_valid_bound_vector_receipt()
        {
            return Err(FreeSemanticError::InvalidManifest);
        }
        Ok(())
    }

    fn bind_staged_receipt(
        &self,
        receipt: &ChannelReceipt,
        channel: &str,
    ) -> Result<ChannelReceipt, FreeSemanticError> {
        let artifact_digest = receipt
            .pending_artifact_digest()
            .filter(|digest| is_sha256_digest(digest))
            .ok_or(FreeSemanticError::InvalidManifest)?;
        Ok(ChannelReceipt::Bound {
            artifact_digest: artifact_digest.to_string(),
            staged_state_digest: staged_channel_state_digest(channel, artifact_digest, self)?,
        })
    }

    fn is_bound_to_staged_state(&self, receipt: &ChannelReceipt, channel: &str) -> bool {
        let Some((artifact_digest, staged_state_digest)) = receipt.bound_parts() else {
            return false;
        };
        is_sha256_digest(artifact_digest)
            && staged_channel_state_digest(channel, artifact_digest, self)
                .is_ok_and(|expected| expected == staged_state_digest)
    }

    fn has_valid_bound_vector_receipt(&self) -> bool {
        match (
            self.semantic_availability.is_available(),
            self.model_descriptor_digest.as_deref(),
            self.dimension,
            self.normalization,
            self.vector_generation,
            self.vector_receipt.as_ref(),
        ) {
            (true, Some(descriptor), Some(dimension), Some(_), Some(generation), Some(receipt)) => {
                generation == self.generation
                    && dimension > 0
                    && dimension <= MAX_VECTOR_DIMENSION
                    && is_sha256_digest(descriptor)
                    && self.is_bound_to_staged_state(receipt, "vector")
            }
            (false, None, None, None, None, None) => true,
            _ => false,
        }
    }
}

fn receipt_digest_fields(receipt: &ChannelReceipt) -> (&str, &str) {
    match receipt {
        ChannelReceipt::Pending { artifact_digest } => (artifact_digest, "pending"),
        ChannelReceipt::Bound {
            artifact_digest,
            staged_state_digest,
        } => (artifact_digest, staged_state_digest),
    }
}

fn staged_channel_state_digest(
    channel: &str,
    artifact_digest: &str,
    manifest: &GenerationManifest,
) -> Result<String, FreeSemanticError> {
    let chunking_policy_digest = manifest
        .chunking_policy_digest
        .as_deref()
        .filter(|digest| is_sha256_digest(digest))
        .ok_or(FreeSemanticError::InvalidManifest)?;
    let chunk_map_digest = manifest
        .chunk_map_digest
        .as_deref()
        .filter(|digest| is_sha256_digest(digest))
        .ok_or(FreeSemanticError::InvalidManifest)?;
    let note_set_digest = manifest
        .note_set_digest
        .as_deref()
        .filter(|digest| is_sha256_digest(digest))
        .ok_or(FreeSemanticError::InvalidManifest)?;
    Ok(sha256_digest(
        "staged-search-channel-receipt-v1",
        &[
            channel,
            artifact_digest,
            &manifest.vault_id,
            &manifest.generation.to_string(),
            manifest
                .model_descriptor_digest
                .as_deref()
                .unwrap_or("none"),
            &manifest
                .dimension
                .map(|dimension| dimension.to_string())
                .unwrap_or_else(|| "none".to_string()),
            &manifest.chunk_format_version.to_string(),
            &manifest.lexical_generation.to_string(),
            &manifest
                .vector_generation
                .map(|generation| generation.to_string())
                .unwrap_or_else(|| "none".to_string()),
            &manifest.rank_policy_version.to_string(),
            &manifest.rank_policy_rrf_k.to_string(),
            &manifest.rank_policy_digest,
            manifest
                .normalization
                .map(VectorNormalization::tag)
                .unwrap_or("none"),
            semantic_availability_tag(manifest.semantic_availability),
            chunking_policy_digest,
            chunk_map_digest,
            note_set_digest,
        ],
    ))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicationToken {
    vault_id: String,
    expected_generation: u64,
    cancelled: bool,
}

impl PublicationToken {
    pub fn expected_generation(&self) -> u64 {
        self.expected_generation
    }

    pub fn cancelled(mut self) -> Self {
        self.cancelled = true;
        self
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct ChunkCatalog {
    vault_id: String,
    chunks: BTreeMap<String, ParagraphChunk>,
    chunks_by_note: BTreeMap<String, BTreeSet<String>>,
    generation: u64,
    manifest: Option<GenerationManifest>,
}

impl fmt::Debug for ChunkCatalog {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChunkCatalog")
            .field("generation", &self.generation)
            .field("chunk_count", &self.chunks.len())
            .field("note_count", &self.chunks_by_note.len())
            .field("has_manifest", &self.manifest.is_some())
            .finish()
    }
}

impl ChunkCatalog {
    pub fn new(vault_id: impl Into<String>) -> Result<Self, FreeSemanticError> {
        Ok(Self {
            vault_id: canonical_identity(vault_id.into(), IdentityKind::Vault)?,
            chunks: BTreeMap::new(),
            chunks_by_note: BTreeMap::new(),
            generation: 0,
            manifest: None,
        })
    }

    pub fn vault_id(&self) -> &str {
        &self.vault_id
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn manifest(&self) -> Option<&GenerationManifest> {
        self.manifest.as_ref()
    }

    pub fn publication_token(&self) -> PublicationToken {
        PublicationToken {
            vault_id: self.vault_id.clone(),
            expected_generation: self.generation,
            cancelled: false,
        }
    }

    pub fn publish(
        &mut self,
        token: PublicationToken,
        projection: ChunkProjection,
        manifest: GenerationManifest,
    ) -> Result<GenerationManifest, FreeSemanticError> {
        self.validate_token(&token)?;
        self.validate_projection(&projection)?;
        let next_generation = self.next_generation()?;
        if let Some(previous) = self.manifest.as_ref() {
            previous.validate_published_state()?;
            if previous.chunking_policy_digest.as_deref()
                != Some(projection.chunking_policy_digest.as_str())
            {
                return Err(FreeSemanticError::ChunkingPolicyMismatch);
            }
        }
        if manifest.vault_id != self.vault_id || manifest.generation != next_generation {
            return Err(FreeSemanticError::ManifestMismatch);
        }
        manifest.validate_shape()?;

        let mut staged_chunks = self.chunks.clone();
        let mut staged_by_note = self.chunks_by_note.clone();
        let projection_note_id = projection.note_id.clone();
        let projection_policy_digest = projection.chunking_policy_digest.clone();
        let projection_chunks = projection.chunks;
        remove_note_from_maps(&mut staged_chunks, &mut staged_by_note, &projection_note_id);
        let mut ids = BTreeSet::new();
        for chunk in projection_chunks {
            ids.insert(chunk.chunk_id.clone());
            staged_chunks.insert(chunk.chunk_id.clone(), chunk);
        }
        if ids.is_empty() {
            staged_by_note.remove(&projection_note_id);
        } else {
            staged_by_note.insert(projection_note_id, ids);
        }
        let manifest = manifest.with_staged_state(
            projection_policy_digest,
            &staged_chunks,
            &staged_by_note,
        )?;
        manifest.validate_published_state()?;

        self.chunks = staged_chunks;
        self.chunks_by_note = staged_by_note;
        self.generation = next_generation;
        self.manifest = Some(manifest.clone());
        Ok(manifest)
    }

    pub fn remove_note(
        &mut self,
        token: PublicationToken,
        note_id: &str,
        lexical_staging: LexicalStagingReceipt,
    ) -> Result<GenerationManifest, FreeSemanticError> {
        self.validate_token(&token)?;
        let note_id = canonical_identity(note_id.to_string(), IdentityKind::Note)?;
        let next_generation = self.next_generation()?;
        let mut staged_chunks = self.chunks.clone();
        let mut staged_by_note = self.chunks_by_note.clone();
        remove_note_from_maps(&mut staged_chunks, &mut staged_by_note, &note_id);
        let receipt = self.stage_degraded_mutation_receipt(
            lexical_staging,
            next_generation,
            &staged_chunks,
            &staged_by_note,
        )?;
        self.chunks = staged_chunks;
        self.chunks_by_note = staged_by_note;
        self.generation = next_generation;
        self.manifest = Some(receipt.clone());
        Ok(receipt)
    }

    pub fn reset(
        &mut self,
        token: PublicationToken,
        lexical_staging: LexicalStagingReceipt,
    ) -> Result<GenerationManifest, FreeSemanticError> {
        self.validate_token(&token)?;
        let next_generation = self.next_generation()?;
        let staged_chunks = BTreeMap::new();
        let staged_by_note = BTreeMap::new();
        let receipt = self.stage_degraded_mutation_receipt(
            lexical_staging,
            next_generation,
            &staged_chunks,
            &staged_by_note,
        )?;
        self.chunks = staged_chunks;
        self.chunks_by_note = staged_by_note;
        self.generation = next_generation;
        self.manifest = Some(receipt.clone());
        Ok(receipt)
    }

    /// A governed contract migration may only happen through a complete,
    /// separately staged replacement. Incremental publish intentionally cannot
    /// mix chunking policies or silently alter a retained note's contract.
    pub fn rebuild(
        &mut self,
        token: PublicationToken,
        chunking_policy: ChunkingPolicy,
        projections: Vec<ChunkProjection>,
        manifest: GenerationManifest,
    ) -> Result<GenerationManifest, FreeSemanticError> {
        self.validate_token(&token)?;
        chunking_policy
            .validate()
            .map_err(|_| FreeSemanticError::InvalidChunkPolicy)?;
        let next_generation = self.next_generation()?;
        if manifest.vault_id != self.vault_id || manifest.generation != next_generation {
            return Err(FreeSemanticError::ManifestMismatch);
        }
        manifest.validate_shape()?;

        let policy_digest = chunking_policy.digest();
        let mut staged_chunks = BTreeMap::new();
        let mut staged_by_note = BTreeMap::new();
        for projection in projections {
            self.validate_projection(&projection)?;
            if projection.chunking_policy != chunking_policy
                || projection.chunking_policy_digest != policy_digest
                || staged_by_note.contains_key(&projection.note_id)
            {
                return Err(FreeSemanticError::ChunkingPolicyMismatch);
            }
            let mut ids = BTreeSet::new();
            for chunk in projection.chunks {
                if !ids.insert(chunk.chunk_id.clone())
                    || staged_chunks
                        .insert(chunk.chunk_id.clone(), chunk)
                        .is_some()
                {
                    return Err(FreeSemanticError::InvalidProjection);
                }
            }
            if !ids.is_empty() {
                staged_by_note.insert(projection.note_id, ids);
            }
        }
        let receipt = manifest.with_staged_state(policy_digest, &staged_chunks, &staged_by_note)?;
        receipt.validate_published_state()?;

        self.chunks = staged_chunks;
        self.chunks_by_note = staged_by_note;
        self.generation = next_generation;
        self.manifest = Some(receipt.clone());
        Ok(receipt)
    }

    pub fn chunk(&self, chunk_id: &str) -> Option<&ParagraphChunk> {
        self.chunks.get(chunk_id)
    }

    pub fn chunks_for_note(&self, note_id: &str) -> Vec<&ParagraphChunk> {
        self.chunks_by_note
            .get(note_id)
            .into_iter()
            .flatten()
            .filter_map(|chunk_id| self.chunks.get(chunk_id))
            .collect()
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Issues an opaque, generation-bound lease for one eventual lexical and
    /// optional semantic execution. Possessing this lease only proves that the
    /// catalog admitted the request; it is not a durable one-use receipt and
    /// does not prove an executor ran.
    pub fn issue_search_lease(
        &self,
        request: &HybridRequest,
        pre_filter_limit: usize,
    ) -> Result<SearchLease, FreeSemanticError> {
        request.validate_for_catalog(&self.vault_id)?;
        if pre_filter_limit < request.limit() || pre_filter_limit > MAX_CHANNEL_HITS {
            return Err(FreeSemanticError::InvalidRequest);
        }
        let manifest = self
            .manifest
            .as_ref()
            .ok_or(FreeSemanticError::InvalidManifest)?;
        manifest.validate_published_state()?;
        Ok(SearchLease::new(
            &self.vault_id,
            self.generation,
            manifest.digest(),
            request,
            pre_filter_limit,
            manifest.lexical_receipt.clone(),
            manifest.vector_receipt.clone(),
        ))
    }

    /// Accepts a structurally bound result assertion from a future channel
    /// adapter. This module deliberately has no executor, so the assertion is
    /// *not* replay-resistant evidence that lexical or vector retrieval ran.
    /// Its counts and completion label are caller assertions, not proof that
    /// an executor considered every eligible upstream row. A real adapter must
    /// replace this boundary with its own durable execution receipt.
    #[allow(clippy::too_many_arguments)]
    pub fn complete_untrusted_channel_assertion(
        &self,
        lease: SearchLease,
        channel: SearchChannel,
        completion: ChannelCompletion,
        upstream_candidate_count: usize,
        excluded_candidate_count: usize,
        hits: Vec<ChannelHit>,
        asserted_title_ids: BTreeSet<String>,
    ) -> Result<ChannelBatch, FreeSemanticError> {
        self.validate_search_lease(&lease)?;
        // Only a future escaped title-field executor can attest exact-title
        // matches. This pure fixture boundary must not turn a copied ID set
        // into a ranking boost.
        if !asserted_title_ids.is_empty() {
            return Err(FreeSemanticError::InvalidChannelBatch);
        }
        let state_receipt = match channel {
            SearchChannel::Lexical => lease.lexical_receipt.clone(),
            SearchChannel::Semantic => lease
                .vector_receipt
                .clone()
                .ok_or(FreeSemanticError::InvalidChannelBatch)?,
        };
        ChannelBatch::from_untrusted_assertion(
            &lease,
            channel,
            state_receipt,
            completion,
            upstream_candidate_count,
            excluded_candidate_count,
            hits,
            |chunk_id| {
                let chunk = self
                    .chunk(chunk_id)
                    .ok_or(FreeSemanticError::InvalidChannelBatch)?;
                if chunk.vault_id != self.vault_id
                    || lease.request.origin_note_id() == Some(chunk.note_id.as_str())
                {
                    return Err(FreeSemanticError::InvalidChannelBatch);
                }
                Ok(())
            },
        )
    }

    pub fn rank_note_hits(
        &self,
        lexical_batch: ChannelBatch,
        semantic_batch: Option<ChannelBatch>,
        request: HybridRequest,
    ) -> Result<Vec<RankedNoteHit>, FreeSemanticError> {
        request.validate_for_catalog(&self.vault_id)?;
        let manifest = self
            .manifest
            .as_ref()
            .ok_or(FreeSemanticError::InvalidManifest)?;
        manifest.validate_published_state()?;
        let rank_policy =
            RankFusionPolicy::new(manifest.rank_policy_version, manifest.rank_policy_rrf_k)
                .map_err(|_| FreeSemanticError::InvalidManifest)?;
        self.validate_bound_batch(
            &lexical_batch,
            SearchChannel::Lexical,
            &manifest.lexical_receipt,
            manifest,
            &request,
        )?;
        let semantic_available = manifest.semantic_availability.is_available();
        let semantic_batch = if semantic_available {
            semantic_batch
                .as_ref()
                .map(|batch| {
                    let receipt = manifest
                        .vector_receipt
                        .as_ref()
                        .ok_or(FreeSemanticError::InvalidManifest)?;
                    self.validate_bound_batch(
                        batch,
                        SearchChannel::Semantic,
                        receipt,
                        manifest,
                        &request,
                    )?;
                    Ok(batch)
                })
                .transpose()?
        } else {
            // A degraded vector channel never gets to influence or invalidate
            // independently verified lexical evidence.
            None
        };
        let mut candidates: BTreeMap<String, Candidate> = BTreeMap::new();
        for hit in &lexical_batch.hits {
            if self.chunk(&hit.chunk_id).is_some() {
                set_best_channel(
                    &mut candidates.entry(hit.chunk_id.clone()).or_default().lexical,
                    hit,
                );
            }
        }
        if let Some(semantic_batch) = semantic_batch {
            for hit in &semantic_batch.hits {
                if self.chunk(&hit.chunk_id).is_some() {
                    set_best_channel(
                        &mut candidates.entry(hit.chunk_id.clone()).or_default().semantic,
                        hit,
                    );
                }
            }
        }

        let mut best_by_note: BTreeMap<String, RankedNoteHit> = BTreeMap::new();
        for (chunk_id, candidate) in candidates {
            let Some(chunk) = self.chunk(&chunk_id) else {
                continue;
            };
            if request.origin_note_id.as_deref() == Some(chunk.note_id.as_str()) {
                continue;
            }
            let ranks = [candidate.lexical.as_ref(), candidate.semantic.as_ref()]
                .into_iter()
                .flatten()
                .map(|hit| hit.rank)
                .collect::<Vec<_>>();
            let rank_score = raw_rrf_rank_score(&ranks, &rank_policy)?;
            let source = match (
                candidate.lexical.is_some(),
                candidate.semantic.is_some(),
                semantic_available,
            ) {
                (true, true, _) => "hybrid",
                (true, false, false) => "lexical-fallback",
                (true, false, true) => "lexical",
                (false, true, _) => "semantic",
                (false, false, _) => continue,
            };
            let hit = RankedNoteHit {
                note_id: chunk.note_id.clone(),
                chunk_id: chunk.chunk_id.clone(),
                title: chunk.title.clone(),
                snippet: bounded_snippet(&chunk.text),
                rank_score,
                source: source.to_string(),
                rank_evidence: RankEvidence {
                    lexical: candidate.lexical,
                    semantic: candidate.semantic,
                    rrf_rank_score: rank_score,
                    // This module receives only untrusted channel assertions.
                    // A real lexical title-field adapter must supply the later
                    // authoritative evidence before title priority can exist.
                    exact_lexical_title: false,
                },
            };
            match best_by_note.get(&hit.note_id) {
                Some(existing) if !is_better_hit(&hit, existing) => {}
                _ => {
                    best_by_note.insert(hit.note_id.clone(), hit);
                }
            }
        }

        let mut hits: Vec<_> = best_by_note.into_values().collect();
        hits.sort_by(|left, right| compare_ranked_hits(left, right));
        hits.truncate(request.limit);
        Ok(hits)
    }

    fn validate_bound_batch(
        &self,
        batch: &ChannelBatch,
        expected_channel: SearchChannel,
        expected_receipt: &ChannelReceipt,
        manifest: &GenerationManifest,
        request: &HybridRequest,
    ) -> Result<(), FreeSemanticError> {
        batch.validate_shape()?;
        if batch.channel != expected_channel
            || batch.vault_id != self.vault_id
            || batch.generation != self.generation
            || batch.manifest_digest != manifest.digest()
            || batch.request_digest != request.digest()
            || batch.state_receipt != *expected_receipt
            || batch.origin_note_id != request.origin_note_id
            || batch.post_filter_limit != request.limit
            || batch.lease_digest
                != search_lease_digest(
                    &self.vault_id,
                    self.generation,
                    &manifest.digest(),
                    request,
                    batch.pre_filter_limit,
                    &manifest.lexical_receipt,
                    manifest.vector_receipt.as_ref(),
                )
            || matches!(batch.completion, ChannelCompletion::Truncated)
                && batch.hits.len() != batch.post_filter_limit
        {
            return Err(FreeSemanticError::InvalidChannelBatch);
        }
        Ok(())
    }

    fn validate_search_lease(&self, lease: &SearchLease) -> Result<(), FreeSemanticError> {
        let manifest = self
            .manifest
            .as_ref()
            .ok_or(FreeSemanticError::InvalidManifest)?;
        manifest.validate_published_state()?;
        lease.validate_for_catalog(&self.vault_id, self.generation, manifest)
    }

    fn stage_degraded_mutation_receipt(
        &self,
        lexical_staging: LexicalStagingReceipt,
        next_generation: u64,
        chunks: &BTreeMap<String, ParagraphChunk>,
        chunks_by_note: &BTreeMap<String, BTreeSet<String>>,
    ) -> Result<GenerationManifest, FreeSemanticError> {
        let previous = self
            .manifest
            .as_ref()
            .ok_or(FreeSemanticError::MissingPublishedReceipt)?;
        previous.validate_published_state()?;
        let chunking_policy_digest = previous
            .chunking_policy_digest
            .clone()
            .ok_or(FreeSemanticError::InvalidManifest)?;
        let manifest = GenerationManifest {
            schema_version: previous.schema_version,
            digest_algorithm: previous.digest_algorithm.clone(),
            vault_id: self.vault_id.clone(),
            generation: next_generation,
            model_descriptor_digest: None,
            dimension: None,
            chunk_format_version: previous.chunk_format_version,
            lexical_generation: next_generation,
            vector_generation: None,
            rank_policy_version: previous.rank_policy_version,
            rank_policy_rrf_k: previous.rank_policy_rrf_k,
            rank_policy_digest: previous.rank_policy_digest.clone(),
            normalization: None,
            semantic_availability: SemanticAvailability::NoCandidateSelected,
            lexical_receipt: lexical_staging.pending_channel_receipt(),
            vector_receipt: None,
            chunking_policy_digest: None,
            chunk_map_digest: None,
            note_set_digest: None,
        };
        manifest.validate_shape()?;
        let receipt = manifest.with_staged_state(chunking_policy_digest, chunks, chunks_by_note)?;
        receipt.validate_published_state()?;
        Ok(receipt)
    }

    fn validate_token(&self, token: &PublicationToken) -> Result<(), FreeSemanticError> {
        if token.cancelled {
            return Err(FreeSemanticError::CancelledPublication);
        }
        if token.vault_id != self.vault_id {
            return Err(FreeSemanticError::VaultMismatch);
        }
        if token.expected_generation != self.generation {
            return Err(FreeSemanticError::StalePublication {
                expected: token.expected_generation,
                actual: self.generation,
            });
        }
        Ok(())
    }

    fn validate_projection(&self, projection: &ChunkProjection) -> Result<(), FreeSemanticError> {
        if projection.vault_id != self.vault_id {
            return Err(FreeSemanticError::VaultMismatch);
        }
        canonical_identity(projection.vault_id.clone(), IdentityKind::Vault)?;
        canonical_identity(projection.note_id.clone(), IdentityKind::Note)?;
        if projection.body_byte_len != projection.canonical_body.len()
            || projection
                .canonical_title
                .len()
                .checked_add(projection.canonical_body.len())
                .map_or(true, |byte_len| byte_len > MAX_NOTE_BYTES)
            || projection.chunks.len() > MAX_CHUNKS_PER_NOTE
            || projection.canonical_title.contains('\0')
            || !is_sha256_digest(&projection.chunking_policy_digest)
        {
            return Err(FreeSemanticError::InvalidProjection);
        }
        let expected_revision = sha256_digest(
            "note-revision-v1",
            &[
                &projection.vault_id,
                &projection.note_id,
                &projection.canonical_title,
                &projection.canonical_body,
            ],
        );
        if projection.note_revision_digest != expected_revision {
            return Err(FreeSemanticError::InvalidProjection);
        }
        projection
            .chunking_policy
            .validate()
            .map_err(|_| FreeSemanticError::InvalidProjection)?;
        if projection.chunking_policy.digest() != projection.chunking_policy_digest {
            return Err(FreeSemanticError::InvalidProjection);
        }
        let canonical_note = NoteInput::new(
            projection.vault_id.clone(),
            projection.note_id.clone(),
            projection.canonical_title.clone(),
            projection.canonical_body.clone(),
        )
        .map_err(|_| FreeSemanticError::InvalidProjection)?;
        let expected_projection = chunk_note(&canonical_note, &projection.chunking_policy)
            .map_err(|_| FreeSemanticError::InvalidProjection)?;
        if &expected_projection != projection {
            return Err(FreeSemanticError::InvalidProjection);
        }
        if projection.coverage.total_body_bytes != projection.body_byte_len
            || projection.coverage.indexed_body_bytes > projection.body_byte_len
        {
            return Err(FreeSemanticError::InvalidProjection);
        }
        let mut multiplicities = BTreeMap::new();
        for chunk in &projection.chunks {
            *multiplicities
                .entry((chunk.kind, chunk.content_digest.clone()))
                .or_insert(0usize) += 1;
        }
        let blocks = markdown_blocks(&projection.canonical_body);
        let mut ids = BTreeSet::new();
        let mut logical_ids = BTreeSet::new();
        let mut expected_occurrences = BTreeMap::new();
        let mut title_count = 0usize;
        for chunk in &projection.chunks {
            let occurrence_key = (chunk.kind, chunk.content_digest.clone());
            let expected_occurrence = *expected_occurrences
                .entry(occurrence_key.clone())
                .or_insert(0usize);
            let ambiguity_revision = multiplicities
                .get(&occurrence_key)
                .is_some_and(|count| *count > 1)
                .then_some(projection.note_revision_digest.as_str());
            let expected_logical_id = logical_chunk_id(
                &projection.vault_id,
                &projection.note_id,
                chunk.kind,
                &chunk.content_digest,
                expected_occurrence,
                ambiguity_revision,
            );
            if chunk.vault_id != self.vault_id
                || chunk.note_id != projection.note_id
                || chunk.title != projection.canonical_title
                || chunk.format_version != CHUNK_FORMAT_VERSION
                || chunk.text.is_empty()
                || chunk.text.len() > MAX_CHUNK_BYTES
                || !ids.insert(chunk.chunk_id.clone())
                || !logical_ids.insert(chunk.logical_id.clone())
                || chunk.content_digest != sha256_digest("chunk-content-v1", &[&chunk.text])
                || chunk.note_revision_digest != projection.note_revision_digest
                || chunk.occurrence != expected_occurrence
                || chunk.logical_id != expected_logical_id
                || chunk.chunk_id != format!("pc{CHUNK_FORMAT_VERSION}:{expected_logical_id}")
            {
                return Err(FreeSemanticError::InvalidProjection);
            }
            match (&chunk.kind, &chunk.body_range) {
                (ChunkKind::Title, None) if chunk.text == projection.canonical_title.trim() => {
                    title_count += 1;
                    if title_count > 1 {
                        return Err(FreeSemanticError::InvalidProjection);
                    }
                }
                (ChunkKind::Title, None) => return Err(FreeSemanticError::InvalidProjection),
                (ChunkKind::Title, Some(_)) | (_, None) => {
                    return Err(FreeSemanticError::InvalidProjection);
                }
                (_, Some(range)) => {
                    if range.start_byte >= range.end_byte
                        || range.end_byte > projection.body_byte_len
                        || !projection.canonical_body.is_char_boundary(range.start_byte)
                        || !projection.canonical_body.is_char_boundary(range.end_byte)
                        || projection.canonical_body[range.start_byte..range.end_byte] != chunk.text
                        || !blocks.iter().any(|block| {
                            block.kind == chunk.kind
                                && range.start_byte >= block.start_byte
                                && range.end_byte <= block.end_byte
                        })
                    {
                        return Err(FreeSemanticError::InvalidProjection);
                    }
                }
            }
            if let Some(occurrence) = expected_occurrences.get_mut(&occurrence_key) {
                *occurrence += 1;
            } else {
                return Err(FreeSemanticError::InvalidProjection);
            }
        }
        let indexed_body_bytes = covered_body_bytes(&projection.chunks);
        if projection.coverage.indexed_body_bytes != indexed_body_bytes {
            return Err(FreeSemanticError::InvalidProjection);
        }
        if let ProjectionStatus::Partial {
            omitted_from_body_byte,
            ..
        } = &projection.coverage.status
        {
            if *omitted_from_body_byte >= projection.body_byte_len
                || indexed_body_bytes > *omitted_from_body_byte
            {
                return Err(FreeSemanticError::InvalidProjection);
            }
        }
        Ok(())
    }

    fn next_generation(&self) -> Result<u64, FreeSemanticError> {
        let next_generation = self
            .generation
            .checked_add(1)
            .ok_or(FreeSemanticError::GenerationOverflow)?;
        Ok(next_generation)
    }
}

fn remove_note_from_maps(
    chunks: &mut BTreeMap<String, ParagraphChunk>,
    chunks_by_note: &mut BTreeMap<String, BTreeSet<String>>,
    note_id: &str,
) {
    if let Some(ids) = chunks_by_note.remove(note_id) {
        for chunk_id in ids {
            chunks.remove(&chunk_id);
        }
    }
}

fn chunk_map_digest(chunks: &BTreeMap<String, ParagraphChunk>) -> String {
    let entry_digests = chunks
        .values()
        .map(|chunk| {
            let occurrence = chunk.occurrence.to_string();
            let range_start = chunk
                .body_range
                .as_ref()
                .map(|range| range.start_byte.to_string())
                .unwrap_or_else(|| "none".to_string());
            let range_end = chunk
                .body_range
                .as_ref()
                .map(|range| range.end_byte.to_string())
                .unwrap_or_else(|| "none".to_string());
            sha256_digest(
                "catalog-chunk-entry-v1",
                &[
                    &chunk.chunk_id,
                    &chunk.logical_id,
                    &chunk.vault_id,
                    &chunk.note_id,
                    &chunk.content_digest,
                    &chunk.note_revision_digest,
                    chunk.kind.tag(),
                    &occurrence,
                    &range_start,
                    &range_end,
                ],
            )
        })
        .collect::<Vec<_>>();
    let fields = entry_digests.iter().map(String::as_str).collect::<Vec<_>>();
    sha256_digest("catalog-chunk-map-v1", &fields)
}

fn note_set_digest(chunks_by_note: &BTreeMap<String, BTreeSet<String>>) -> String {
    let note_digests = chunks_by_note
        .iter()
        .map(|(note_id, chunk_ids)| {
            let fields = chunk_ids.iter().map(String::as_str).collect::<Vec<_>>();
            let chunk_ids_digest = sha256_digest("catalog-note-chunk-ids-v1", &fields);
            sha256_digest("catalog-note-set-entry-v1", &[note_id, &chunk_ids_digest])
        })
        .collect::<Vec<_>>();
    let fields = note_digests.iter().map(String::as_str).collect::<Vec<_>>();
    sha256_digest("catalog-note-set-v1", &fields)
}

#[derive(Default)]
struct Candidate {
    lexical: Option<ChannelHit>,
    semantic: Option<ChannelHit>,
}

fn validate_channel_hits(
    hits: &[ChannelHit],
    validate_entries: bool,
) -> Result<(), FreeSemanticError> {
    if hits.len() > MAX_CHANNEL_HITS {
        return Err(FreeSemanticError::TooManyChannelHits {
            limit: MAX_CHANNEL_HITS,
        });
    }
    if !validate_entries {
        return Ok(());
    }

    let mut chunk_ids = BTreeSet::new();
    let mut ranks = BTreeSet::new();
    for hit in hits {
        hit.validate()?;
        if !chunk_ids.insert(hit.chunk_id.clone()) {
            return Err(FreeSemanticError::DuplicateChannelHit);
        }
        if !ranks.insert(hit.rank) {
            return Err(FreeSemanticError::DuplicateChannelRank);
        }
    }
    if ranks
        .iter()
        .enumerate()
        .any(|(index, rank)| *rank != index + 1)
    {
        return Err(FreeSemanticError::InvalidChannelHit);
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChannelHit {
    pub chunk_id: String,
    /// Native score from one channel. It is not comparable to another channel
    /// and is never exposed as a probability or calibrated confidence.
    pub raw_score: f32,
    pub rank: usize,
}

impl ChannelHit {
    pub fn new(chunk_id: String, raw_score: f32, rank: usize) -> Result<Self, FreeSemanticError> {
        let hit = Self {
            chunk_id,
            raw_score,
            rank,
        };
        hit.validate()?;
        Ok(hit)
    }

    fn validate(&self) -> Result<(), FreeSemanticError> {
        if !is_canonical_chunk_id(&self.chunk_id)
            || !self.raw_score.is_finite()
            || self.rank == 0
            || self.rank > MAX_CHANNEL_HITS
        {
            return Err(FreeSemanticError::InvalidChannelHit);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VectorFixture {
    pub chunk_id: String,
    pub vector: Vec<f32>,
}

/// Deterministic injected-vector ranking for source fixtures and future
/// adapter tests. It returns raw cosine similarity and rank, never a made-up
/// confidence transform.
pub fn cosine_search(
    query: &[f32],
    vectors: &[VectorFixture],
    limit: usize,
) -> Result<Vec<ChannelHit>, FreeSemanticError> {
    validate_vector(query)?;
    if vectors.len() > MAX_VECTOR_COUNT {
        return Err(FreeSemanticError::TooManyVectors {
            limit: MAX_VECTOR_COUNT,
        });
    }
    if limit == 0 {
        return Ok(Vec::new());
    }
    if limit > MAX_CHANNEL_HITS {
        return Err(FreeSemanticError::InvalidRequest);
    }
    let query_norm = l2_norm(query)?;
    let mut scored = Vec::with_capacity(vectors.len());
    let mut chunk_ids = BTreeSet::new();
    for fixture in vectors {
        if !is_canonical_chunk_id(&fixture.chunk_id)
            || !chunk_ids.insert(fixture.chunk_id.clone())
            || fixture.vector.len() != query.len()
        {
            return Err(FreeSemanticError::InvalidVector);
        }
        let vector_norm = l2_norm(&fixture.vector)?;
        let dot = dot(query, &fixture.vector)?;
        let denominator = query_norm * vector_norm;
        let cosine = dot / denominator;
        if !cosine.is_finite() {
            return Err(FreeSemanticError::InvalidVector);
        }
        scored.push((fixture.chunk_id.clone(), cosine.clamp(-1.0, 1.0) as f32));
    }
    scored.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    scored.truncate(limit);
    scored
        .into_iter()
        .enumerate()
        .map(|(index, (chunk_id, raw_score))| ChannelHit::new(chunk_id, raw_score, index + 1))
        .collect()
}

fn validate_vector(vector: &[f32]) -> Result<(), FreeSemanticError> {
    if vector.is_empty()
        || vector.len() > MAX_VECTOR_DIMENSION
        || vector.iter().any(|value| !value.is_finite())
    {
        return Err(FreeSemanticError::InvalidVector);
    }
    Ok(())
}

fn dot(left: &[f32], right: &[f32]) -> Result<f64, FreeSemanticError> {
    if left.len() != right.len() {
        return Err(FreeSemanticError::InvalidVector);
    }
    let value = left
        .iter()
        .zip(right)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum::<f64>();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FreeSemanticError::InvalidVector)
    }
}

fn l2_norm(vector: &[f32]) -> Result<f64, FreeSemanticError> {
    validate_vector(vector)?;
    let squared = dot(vector, vector)?;
    if squared <= FLOAT_EPSILON {
        return Err(FreeSemanticError::InvalidVector);
    }
    let norm = squared.sqrt();
    if norm.is_finite() {
        Ok(norm)
    } else {
        Err(FreeSemanticError::InvalidVector)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticAvailability {
    Available,
    NoCandidateSelected,
    MissingAsset,
    CorruptAsset,
    Rebuilding,
    Cancelled,
    DimensionMismatch,
}

impl SemanticAvailability {
    fn is_available(self) -> bool {
        self == Self::Available
    }
}

fn semantic_availability_tag(value: SemanticAvailability) -> &'static str {
    match value {
        SemanticAvailability::Available => "available",
        SemanticAvailability::NoCandidateSelected => "no-candidate-selected",
        SemanticAvailability::MissingAsset => "missing-asset",
        SemanticAvailability::CorruptAsset => "corrupt-asset",
        SemanticAvailability::Rebuilding => "rebuilding",
        SemanticAvailability::Cancelled => "cancelled",
        SemanticAvailability::DimensionMismatch => "dimension-mismatch",
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankFusionPolicy {
    pub version: u32,
    pub rrf_k: usize,
}

impl RankFusionPolicy {
    pub fn new(version: u32, rrf_k: usize) -> Result<Self, FreeSemanticError> {
        let policy = Self { version, rrf_k };
        policy.validate()?;
        Ok(policy)
    }

    fn validate(&self) -> Result<(), FreeSemanticError> {
        if self.version == 0 || self.rrf_k == 0 || self.rrf_k > 10_000 {
            return Err(FreeSemanticError::InvalidRankPolicy);
        }
        Ok(())
    }

    fn digest(&self) -> String {
        sha256_digest(
            "rank-fusion-policy-v1",
            &[&self.version.to_string(), &self.rrf_k.to_string()],
        )
    }
}

/// A raw rank-fusion signal. It is intentionally not a calibrated confidence,
/// probability, or cross-channel score normalization.
pub fn raw_rrf_rank_score(
    ranks: &[usize],
    policy: &RankFusionPolicy,
) -> Result<f64, FreeSemanticError> {
    policy.validate()?;
    if ranks.is_empty() || ranks.len() > 2 {
        return Err(FreeSemanticError::InvalidChannelHit);
    }
    let mut total = 0.0;
    for rank in ranks {
        if *rank == 0 || *rank > MAX_CHANNEL_HITS {
            return Err(FreeSemanticError::InvalidChannelHit);
        }
        total += 1.0 / (policy.rrf_k as f64 + *rank as f64);
    }
    if total.is_finite() {
        Ok(total)
    } else {
        Err(FreeSemanticError::InvalidRankPolicy)
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct HybridRequest {
    vault_id: String,
    query: String,
    limit: usize,
    origin_note_id: Option<String>,
    exact_title: Option<String>,
}

impl fmt::Debug for HybridRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HybridRequest")
            .field("format_version", &SEARCH_REQUEST_FORMAT_VERSION)
            .field("query_bytes", &self.query.len())
            .field("limit", &self.limit)
            .field("has_origin_note", &self.origin_note_id.is_some())
            .field("has_exact_title", &self.exact_title.is_some())
            .finish()
    }
}

impl HybridRequest {
    pub fn new(
        vault_id: impl Into<String>,
        query: impl Into<String>,
        limit: usize,
        origin_note_id: Option<String>,
    ) -> Result<Self, FreeSemanticError> {
        let request = Self {
            vault_id: canonical_identity(vault_id.into(), IdentityKind::Vault)?,
            query: query.into(),
            limit,
            origin_note_id: origin_note_id
                .map(|origin| canonical_identity(origin, IdentityKind::Note))
                .transpose()?,
            exact_title: None,
        };
        request.validate_shape()?;
        Ok(request)
    }

    pub fn with_exact_title(mut self, title: impl Into<String>) -> Result<Self, FreeSemanticError> {
        let title = title.into().trim().to_string();
        if title.is_empty() {
            self.exact_title = None;
        } else if title.len() > MAX_EXACT_TITLE_BYTES || title.chars().any(char::is_control) {
            return Err(FreeSemanticError::InvalidRequest);
        } else {
            self.exact_title = Some(title);
        }
        self.validate_shape()?;
        Ok(self)
    }

    pub fn digest(&self) -> String {
        sha256_digest(
            "note-recall-request-v2",
            &[
                &SEARCH_REQUEST_FORMAT_VERSION.to_string(),
                &self.vault_id,
                &self.query_digest(),
                LEXICAL_INPUT_POLICY,
                &LEXICAL_INPUT_POLICY_VERSION.to_string(),
                SEMANTIC_INPUT_POLICY,
                &SEMANTIC_INPUT_POLICY_VERSION.to_string(),
                &self.limit.to_string(),
                self.origin_note_id.as_deref().unwrap_or("none"),
                self.exact_title.as_deref().unwrap_or("none"),
            ],
        )
    }

    /// Digest exact bounded UTF-8 query bytes without exposing them to logs or
    /// generic callers. Future adapters must bind any derived query to this.
    pub fn query_digest(&self) -> String {
        sha256_digest("note-recall-query-bytes-v1", &[&self.query])
    }

    pub fn limit(&self) -> usize {
        self.limit
    }

    pub fn origin_note_id(&self) -> Option<&str> {
        self.origin_note_id.as_deref()
    }

    fn validate_for_catalog(&self, vault_id: &str) -> Result<(), FreeSemanticError> {
        self.validate_shape()?;
        if self.vault_id != vault_id {
            return Err(FreeSemanticError::VaultMismatch);
        }
        Ok(())
    }

    fn validate_shape(&self) -> Result<(), FreeSemanticError> {
        canonical_identity(self.vault_id.clone(), IdentityKind::Vault)?;
        if self.query.is_empty()
            || self.query.trim().is_empty()
            || self.query.len() > MAX_QUERY_BYTES
            || self.query.chars().any(char::is_control)
        {
            return Err(FreeSemanticError::InvalidRequest);
        }
        if self.limit == 0 || self.limit > MAX_QUERY_LIMIT {
            return Err(FreeSemanticError::InvalidRequest);
        }
        if let Some(origin_note_id) = &self.origin_note_id {
            canonical_identity(origin_note_id.clone(), IdentityKind::Note)?;
        }
        if let Some(exact_title) = &self.exact_title {
            if exact_title.is_empty()
                || exact_title.len() > MAX_EXACT_TITLE_BYTES
                || exact_title.chars().any(char::is_control)
            {
                return Err(FreeSemanticError::InvalidRequest);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchChannel {
    Lexical,
    Semantic,
}

impl SearchChannel {
    fn tag(self) -> &'static str {
        match self {
            Self::Lexical => "lexical",
            Self::Semantic => "semantic",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelCompletion {
    Complete,
    Truncated,
}

/// Opaque request/generation admission issued by `ChunkCatalog`. Its fields
/// remain private so a caller cannot hand-assemble a current-looking receipt.
#[derive(PartialEq, Eq)]
pub struct SearchLease {
    vault_id: String,
    generation: u64,
    manifest_digest: String,
    request: HybridRequest,
    pre_filter_limit: usize,
    lexical_receipt: ChannelReceipt,
    vector_receipt: Option<ChannelReceipt>,
    lease_digest: String,
}

impl fmt::Debug for SearchLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SearchLease")
            .field("generation", &self.generation)
            .field("pre_filter_limit", &self.pre_filter_limit)
            .field("semantic_channel_admitted", &self.vector_receipt.is_some())
            .finish()
    }
}

impl SearchLease {
    fn new(
        vault_id: &str,
        generation: u64,
        manifest_digest: String,
        request: &HybridRequest,
        pre_filter_limit: usize,
        lexical_receipt: ChannelReceipt,
        vector_receipt: Option<ChannelReceipt>,
    ) -> Self {
        let lease_digest = search_lease_digest(
            vault_id,
            generation,
            &manifest_digest,
            request,
            pre_filter_limit,
            &lexical_receipt,
            vector_receipt.as_ref(),
        );
        Self {
            vault_id: vault_id.to_string(),
            generation,
            manifest_digest,
            request: request.clone(),
            pre_filter_limit,
            lexical_receipt,
            vector_receipt,
            lease_digest,
        }
    }

    fn validate_for_catalog(
        &self,
        vault_id: &str,
        generation: u64,
        manifest: &GenerationManifest,
    ) -> Result<(), FreeSemanticError> {
        if self.vault_id != vault_id
            || self.generation != generation
            || self.manifest_digest != manifest.digest()
            || self.pre_filter_limit < self.request.limit()
            || self.pre_filter_limit > MAX_CHANNEL_HITS
            || self.lexical_receipt != manifest.lexical_receipt
            || self.vector_receipt != manifest.vector_receipt
            || self.lease_digest
                != search_lease_digest(
                    &self.vault_id,
                    self.generation,
                    &self.manifest_digest,
                    &self.request,
                    self.pre_filter_limit,
                    &self.lexical_receipt,
                    self.vector_receipt.as_ref(),
                )
        {
            return Err(FreeSemanticError::InvalidChannelBatch);
        }
        Ok(())
    }
}

fn search_lease_digest(
    vault_id: &str,
    generation: u64,
    manifest_digest: &str,
    request: &HybridRequest,
    pre_filter_limit: usize,
    lexical_receipt: &ChannelReceipt,
    vector_receipt: Option<&ChannelReceipt>,
) -> String {
    let (lexical_artifact, lexical_state) = receipt_digest_fields(lexical_receipt);
    let (vector_artifact, vector_state) = vector_receipt
        .map(receipt_digest_fields)
        .unwrap_or(("none", "none"));
    sha256_digest(
        "search-lease-v1",
        &[
            vault_id,
            &generation.to_string(),
            manifest_digest,
            &request.digest(),
            &pre_filter_limit.to_string(),
            lexical_artifact,
            lexical_state,
            vector_artifact,
            vector_state,
        ],
    )
}

#[derive(Debug, PartialEq)]
pub struct ChannelBatch {
    channel: SearchChannel,
    vault_id: String,
    generation: u64,
    manifest_digest: String,
    request_digest: String,
    state_receipt: ChannelReceipt,
    origin_note_id: Option<String>,
    pre_filter_limit: usize,
    post_filter_limit: usize,
    completion: ChannelCompletion,
    upstream_candidate_count: usize,
    excluded_candidate_count: usize,
    lease_digest: String,
    result_digest: String,
    hits: Vec<ChannelHit>,
}

impl ChannelBatch {
    #[allow(clippy::too_many_arguments)]
    fn from_untrusted_assertion(
        lease: &SearchLease,
        channel: SearchChannel,
        state_receipt: ChannelReceipt,
        completion: ChannelCompletion,
        upstream_candidate_count: usize,
        excluded_candidate_count: usize,
        hits: Vec<ChannelHit>,
        validate_hit: impl Fn(&str) -> Result<(), FreeSemanticError>,
    ) -> Result<Self, FreeSemanticError> {
        let batch = Self {
            channel,
            vault_id: lease.vault_id.clone(),
            generation: lease.generation,
            manifest_digest: lease.manifest_digest.clone(),
            request_digest: lease.request.digest(),
            state_receipt,
            origin_note_id: lease.request.origin_note_id().map(str::to_string),
            pre_filter_limit: lease.pre_filter_limit,
            post_filter_limit: lease.request.limit(),
            completion,
            upstream_candidate_count,
            excluded_candidate_count,
            lease_digest: lease.lease_digest.clone(),
            result_digest: String::new(),
            hits,
        };
        batch.validate_shape()?;
        for hit in &batch.hits {
            validate_hit(&hit.chunk_id)?;
        }
        let mut batch = batch;
        batch.result_digest = batch.computed_result_digest();
        Ok(batch)
    }

    fn validate_shape(&self) -> Result<(), FreeSemanticError> {
        if self.generation == 0
            || !is_sha256_digest(&self.manifest_digest)
            || !is_sha256_digest(&self.request_digest)
            || self.pre_filter_limit == 0
            || self.pre_filter_limit > MAX_CHANNEL_HITS
            || self.post_filter_limit == 0
            || self.post_filter_limit > self.pre_filter_limit
            || self.upstream_candidate_count > MAX_CHANNEL_HITS
            || self.excluded_candidate_count > self.upstream_candidate_count
            || self.hits.len() > self.post_filter_limit
        {
            return Err(FreeSemanticError::InvalidChannelBatch);
        }
        validate_channel_hits(&self.hits, true)?;
        let eligible_candidate_count = self
            .upstream_candidate_count
            .checked_sub(self.excluded_candidate_count)
            .ok_or(FreeSemanticError::InvalidChannelBatch)?;
        match self.completion {
            ChannelCompletion::Complete
                if self.hits.len() != eligible_candidate_count.min(self.post_filter_limit) =>
            {
                return Err(FreeSemanticError::InvalidChannelBatch);
            }
            ChannelCompletion::Truncated
                if self.upstream_candidate_count < self.pre_filter_limit
                    || self.hits.len() != self.post_filter_limit =>
            {
                return Err(FreeSemanticError::InvalidChannelBatch);
            }
            _ => {}
        }
        if !self.result_digest.is_empty() && self.result_digest != self.computed_result_digest() {
            return Err(FreeSemanticError::InvalidChannelBatch);
        }
        Ok(())
    }

    fn computed_result_digest(&self) -> String {
        let hit_digests = self
            .hits
            .iter()
            .map(|hit| {
                sha256_digest(
                    "channel-hit-v1",
                    &[
                        &hit.chunk_id,
                        &hit.raw_score.to_bits().to_string(),
                        &hit.rank.to_string(),
                    ],
                )
            })
            .collect::<Vec<_>>();
        let hit_fields = hit_digests.iter().map(String::as_str).collect::<Vec<_>>();
        let hits_digest = sha256_digest("channel-batch-hits-v1", &hit_fields);
        sha256_digest(
            "untrusted-channel-batch-assertion-v1",
            &[
                self.channel.tag(),
                &self.vault_id,
                &self.generation.to_string(),
                &self.manifest_digest,
                &self.request_digest,
                &self.pre_filter_limit.to_string(),
                &self.post_filter_limit.to_string(),
                match self.completion {
                    ChannelCompletion::Complete => "complete",
                    ChannelCompletion::Truncated => "truncated",
                },
                &self.upstream_candidate_count.to_string(),
                &self.excluded_candidate_count.to_string(),
                &self.lease_digest,
                &hits_digest,
            ],
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RankEvidence {
    pub lexical: Option<ChannelHit>,
    pub semantic: Option<ChannelHit>,
    pub rrf_rank_score: f64,
    pub exact_lexical_title: bool,
}

#[derive(Clone, PartialEq)]
pub struct RankedNoteHit {
    pub note_id: String,
    pub chunk_id: String,
    pub title: String,
    pub snippet: String,
    /// Ordering-only rank signal. Never render or threshold this as a
    /// probability until a measured corpus calibrates a display contract.
    pub rank_score: f64,
    pub source: String,
    pub rank_evidence: RankEvidence,
}

impl fmt::Debug for RankedNoteHit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RankedNoteHit")
            .field("rank_score", &self.rank_score)
            .field("has_source_label", &!self.source.is_empty())
            .field(
                "has_lexical_evidence",
                &self.rank_evidence.lexical.is_some(),
            )
            .field(
                "has_semantic_evidence",
                &self.rank_evidence.semantic.is_some(),
            )
            .finish()
    }
}

fn set_best_channel(slot: &mut Option<ChannelHit>, candidate: &ChannelHit) {
    let should_replace = match slot {
        None => true,
        Some(existing) => {
            candidate.rank < existing.rank
                || (candidate.rank == existing.rank
                    && candidate.raw_score.total_cmp(&existing.raw_score).is_gt())
                || (candidate.rank == existing.rank
                    && candidate.raw_score.total_cmp(&existing.raw_score).is_eq()
                    && candidate.chunk_id < existing.chunk_id)
        }
    };
    if should_replace {
        *slot = Some(candidate.clone());
    }
}

fn is_better_hit(candidate: &RankedNoteHit, existing: &RankedNoteHit) -> bool {
    compare_ranked_hits(candidate, existing).is_lt()
}

fn compare_ranked_hits(left: &RankedNoteHit, right: &RankedNoteHit) -> std::cmp::Ordering {
    right
        .rank_evidence
        .exact_lexical_title
        .cmp(&left.rank_evidence.exact_lexical_title)
        .then_with(|| right.rank_score.total_cmp(&left.rank_score))
        .then_with(|| left.note_id.cmp(&right.note_id))
        .then_with(|| left.chunk_id.cmp(&right.chunk_id))
}

fn bounded_snippet(text: &str) -> String {
    const MAX_BYTES: usize = 160;
    if text.len() <= MAX_BYTES {
        return text.to_string();
    }
    let end_byte = previous_char_boundary(text, MAX_BYTES);
    format!("{}…", &text[..end_byte])
}

/// SHA-256 with domain separation and length framing prevents ambiguous field
/// concatenation while keeping every content/manifest comparison collision-safe.
pub fn sha256_digest(domain: &str, fields: &[&str]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"epistemos-free-semantic\0");
    update_framed(&mut hasher, domain.as_bytes());
    for field in fields {
        update_framed(&mut hasher, field.as_bytes());
    }
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity("sha256:".len() + digest.len() * 2);
    encoded.push_str("sha256:");
    for byte in digest {
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

fn update_framed(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

fn is_sha256_digest(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn is_canonical_chunk_id(value: &str) -> bool {
    if value.len() > MAX_ID_BYTES || value.chars().any(char::is_control) {
        return false;
    }
    let prefix = format!("pc{CHUNK_FORMAT_VERSION}:");
    value.strip_prefix(&prefix).is_some_and(is_sha256_digest)
}

#[derive(Clone, Copy)]
enum IdentityKind {
    Vault,
    Note,
}

fn canonical_identity(value: String, kind: IdentityKind) -> Result<String, FreeSemanticError> {
    if value.is_empty() {
        return Err(match kind {
            IdentityKind::Vault => FreeSemanticError::EmptyVaultID,
            IdentityKind::Note => FreeSemanticError::EmptyNoteID,
        });
    }
    if value.trim() != value.as_str()
        || value.len() > MAX_ID_BYTES
        || value.chars().any(char::is_control)
        || value.contains('/')
        || value.contains('\\')
        || matches!(value.as_str(), "." | "..")
    {
        return Err(match kind {
            IdentityKind::Vault => FreeSemanticError::InvalidVaultID,
            IdentityKind::Note => FreeSemanticError::InvalidNoteID,
        });
    }
    Ok(value)
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum FreeSemanticError {
    #[error("vault identifier was empty")]
    EmptyVaultID,
    #[error("note identifier was empty")]
    EmptyNoteID,
    #[error("vault identifier was malformed or non-canonical")]
    InvalidVaultID,
    #[error("note identifier was malformed or non-canonical")]
    InvalidNoteID,
    #[error("note exceeds bounded byte limit {limit}")]
    NoteTooLarge { limit: usize },
    #[error("chunk policy exceeds the Free search bounds")]
    InvalidChunkPolicy,
    #[error("chunk exceeds bounded byte limit {limit}")]
    ChunkExceedsPolicy { limit: usize },
    #[error("chunk body range was invalid")]
    InvalidBodyRange,
    #[error("incremental publication changed the catalog-wide chunking policy")]
    ChunkingPolicyMismatch,
    #[error("projection failed invariant validation")]
    InvalidProjection,
    #[error("generation manifest failed invariant validation")]
    InvalidManifest,
    #[error("generation manifest did not match the requested publication")]
    ManifestMismatch,
    #[error("a mutation requires the prior published generation receipt")]
    MissingPublishedReceipt,
    #[error("publication was cancelled before it could publish")]
    CancelledPublication,
    #[error("publication generation was stale: expected {expected}, actual {actual}")]
    StalePublication { expected: u64, actual: u64 },
    #[error("publication generation overflow requires a bounded rebuild")]
    GenerationOverflow,
    #[error("chunk occurrence overflow requires a bounded rebuild")]
    ChunkOccurrenceOverflow,
    #[error("vault identity did not match the current catalog")]
    VaultMismatch,
    #[error("vector was empty, non-finite, zero-norm, over-bound, or dimension-mismatched")]
    InvalidVector,
    #[error("vector collection exceeds bounded count {limit}")]
    TooManyVectors { limit: usize },
    #[error("channel hit was non-finite, rankless, or lacked a chunk identity")]
    InvalidChannelHit,
    #[error("channel hit collection exceeds bounded count {limit}")]
    TooManyChannelHits { limit: usize },
    #[error("channel included a duplicate chunk identity")]
    DuplicateChannelHit,
    #[error("channel included a duplicate rank")]
    DuplicateChannelRank,
    #[error("channel batch did not bind one bounded request and published receipt")]
    InvalidChannelBatch,
    #[error("request exceeded the bounded note-search contract")]
    InvalidRequest,
    #[error("rank fusion policy was invalid")]
    InvalidRankPolicy,
}

#[cfg(test)]
mod generation_contract_tests {
    use super::*;

    #[test]
    fn maximum_generation_rejects_a_mutation_without_changing_catalog_state() {
        let mut catalog = ChunkCatalog {
            vault_id: "vault-a".into(),
            chunks: BTreeMap::new(),
            chunks_by_note: BTreeMap::new(),
            generation: u64::MAX,
            manifest: None,
        };
        let before = catalog.clone();
        let manifest = GenerationManifest::new(
            "vault-a",
            1,
            Some("sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into()),
            Some(256),
            CHUNK_FORMAT_VERSION,
            1,
            None,
            RankFusionPolicy::new(1, 60).unwrap(),
            Some(VectorNormalization::InjectedCosineL2),
            SemanticAvailability::MissingAsset,
            "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            None,
        )
        .unwrap();

        assert_eq!(
            catalog.reset(catalog.publication_token(), manifest),
            Err(FreeSemanticError::GenerationOverflow)
        );
        assert_eq!(catalog, before);
    }
}
