use epistemos_shadow::backend::free_semantic::{
    CHUNK_FORMAT_VERSION, ChannelBatch, ChannelCompletion, ChannelHit, ChannelReceipt,
    ChunkCatalog, ChunkKind, ChunkingPolicy, GenerationManifest, HybridRequest,
    LexicalStagingReceipt, NoteInput, ProjectionPartialReason, ProjectionStatus, RankFusionPolicy,
    SearchChannel, SemanticAvailability, VectorFixture, VectorNormalization, chunk_note,
    cosine_search, raw_rrf_rank_score,
};
use std::collections::BTreeSet;

fn note(vault_id: &str, note_id: &str, title: &str, body: &str) -> NoteInput {
    NoteInput::new(vault_id, note_id, title, body).expect("note fixture")
}

fn policy() -> RankFusionPolicy {
    RankFusionPolicy::new(1, 60).expect("rank policy")
}

fn request(limit: usize, origin_note_id: Option<&str>) -> HybridRequest {
    request_with_query("natural paragraph query", limit, origin_note_id)
}

fn request_with_query(query: &str, limit: usize, origin_note_id: Option<&str>) -> HybridRequest {
    HybridRequest::new("vault-a", query, limit, origin_note_id.map(str::to_string))
        .expect("bounded vault-scoped request")
}

fn manifest(vault_id: &str, generation: u64) -> GenerationManifest {
    manifest_with_availability(
        vault_id,
        generation,
        SemanticAvailability::NoCandidateSelected,
    )
}

fn semantic_manifest(vault_id: &str, generation: u64) -> GenerationManifest {
    manifest_with_availability(vault_id, generation, SemanticAvailability::Available)
}

fn manifest_with_availability(
    vault_id: &str,
    generation: u64,
    semantic_availability: SemanticAvailability,
) -> GenerationManifest {
    GenerationManifest::new(
        vault_id,
        generation,
        (semantic_availability == SemanticAvailability::Available).then_some(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
        ),
        (semantic_availability == SemanticAvailability::Available).then_some(256),
        CHUNK_FORMAT_VERSION,
        generation,
        (semantic_availability == SemanticAvailability::Available).then_some(generation),
        policy(),
        (semantic_availability == SemanticAvailability::Available)
            .then_some(VectorNormalization::InjectedCosineL2),
        semantic_availability,
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        (semantic_availability == SemanticAvailability::Available).then_some(
            "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".into(),
        ),
    )
    .expect("manifest fixture")
}

fn publish(
    catalog: &mut ChunkCatalog,
    projection: &epistemos_shadow::backend::free_semantic::ChunkProjection,
) {
    let token = catalog.publication_token();
    let next_generation = token.expected_generation() + 1;
    catalog
        .publish(
            token,
            projection.clone(),
            manifest(catalog.vault_id(), next_generation),
        )
        .expect("atomic fixture publish");
}

fn publish_semantic(
    catalog: &mut ChunkCatalog,
    projection: &epistemos_shadow::backend::free_semantic::ChunkProjection,
) {
    let token = catalog.publication_token();
    let next_generation = token.expected_generation() + 1;
    catalog
        .publish(
            token,
            projection.clone(),
            semantic_manifest(catalog.vault_id(), next_generation),
        )
        .expect("atomic semantic fixture publish");
}

fn semantic_hits(query: &[f32], chunks: &[(String, Vec<f32>)]) -> Vec<ChannelHit> {
    let fixtures: Vec<VectorFixture> = chunks
        .iter()
        .map(|(chunk_id, vector)| VectorFixture {
            chunk_id: chunk_id.clone(),
            vector: vector.clone(),
        })
        .collect();
    cosine_search(query, &fixtures, 32).expect("well-formed injected vectors")
}

fn lexical_staging() -> LexicalStagingReceipt {
    LexicalStagingReceipt::new(
        "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
    )
    .expect("bounded lexical staging assertion")
}

fn batch(
    catalog: &ChunkCatalog,
    request: &HybridRequest,
    channel: SearchChannel,
    hits: Vec<ChannelHit>,
    exact_title_chunk_ids: impl IntoIterator<Item = String>,
) -> ChannelBatch {
    try_batch(catalog, request, channel, hits, exact_title_chunk_ids).expect("bound channel batch")
}

fn try_batch(
    catalog: &ChunkCatalog,
    request: &HybridRequest,
    channel: SearchChannel,
    hits: Vec<ChannelHit>,
    exact_title_chunk_ids: impl IntoIterator<Item = String>,
) -> Result<ChannelBatch, epistemos_shadow::backend::free_semantic::FreeSemanticError> {
    let lease = catalog.issue_search_lease(request, 32)?;
    catalog.complete_untrusted_channel_assertion(
        lease,
        channel,
        ChannelCompletion::Complete,
        hits.len(),
        0,
        hits,
        exact_title_chunk_ids.into_iter().collect::<BTreeSet<_>>(),
    )
}

#[test]
fn raw_rank_signal_stays_distinct_from_a_visibility_confidence() {
    let raw = raw_rrf_rank_score(&[1, 1], &policy()).expect("rank signal");
    assert!(raw < 0.2, "documents the old raw-RRF/Halo mismatch");
    assert!(raw.is_finite());
}

#[test]
fn checksum_dependency_and_generation_advance_are_explicit_source_contracts() {
    let manifest = include_str!("../Cargo.toml");
    let module = include_str!("../src/backend/free_semantic.rs");
    assert!(manifest.contains("sha2 = \"0.10\""));
    assert!(module.contains("checked_add(1)"));
    assert!(
        module
            .contains("let projection_policy_digest = projection.chunking_policy_digest.clone();")
    );
    assert!(!module.contains("for chunk in projection.chunks {\n            ids.insert"));
    assert!(module.contains("let lexical_pending = self.lexical_receipt.clone();"));
}

#[test]
fn paragraph_vectors_return_the_matching_paragraph_not_a_diluted_whole_note() {
    let projection = chunk_note(
        &note(
            "vault-a",
            "n-astronomy",
            "Field notes",
            "A long unrelated preface about receipts and schedules.\n\n## Astronomy\n\nThe observatory opens before dawn for the meteor survey.",
        ),
        &ChunkingPolicy::default(),
    )
    .expect("bounded Unicode-safe chunking");
    let mut catalog = ChunkCatalog::new("vault-a").expect("catalog");
    publish_semantic(&mut catalog, &projection);

    let semantic = semantic_hits(
        &[1.0, 0.0],
        &projection
            .chunks
            .iter()
            .map(|chunk| {
                let vector = if chunk.text.contains("observatory") {
                    vec![1.0, 0.0]
                } else {
                    vec![-1.0, 0.0]
                };
                (chunk.chunk_id.clone(), vector)
            })
            .collect::<Vec<_>>(),
    );

    let request = request(5, None);
    let hits = catalog
        .rank_note_hits(
            batch(&catalog, &request, SearchChannel::Lexical, vec![], []),
            Some(batch(
                &catalog,
                &request,
                SearchChannel::Semantic,
                semantic,
                [],
            )),
            request,
        )
        .expect("ranked hits");
    assert_eq!(hits.len(), 1);
    assert!(hits[0].snippet.contains("observatory"));
    assert_eq!(hits[0].source, "semantic");
    assert!(hits[0].rank_evidence.semantic.is_some());
    assert!(hits[0].rank_evidence.lexical.is_none());
}

#[test]
fn untrusted_title_assertions_are_rejected_and_cannot_change_ranking() {
    let exact = chunk_note(
        &note("vault-a", "z-exact", "Asterism ZX-81", "A concise note."),
        &ChunkingPolicy::default(),
    )
    .expect("exact chunks");
    let related = chunk_note(
        &note(
            "vault-a",
            "a-related",
            "Night sky",
            "A semantically similar note.",
        ),
        &ChunkingPolicy::default(),
    )
    .expect("related chunks");
    let mut catalog = ChunkCatalog::new("vault-a").expect("catalog");
    publish_semantic(&mut catalog, &related);
    publish_semantic(&mut catalog, &exact);

    let request = request(5, None)
        .with_exact_title("Asterism ZX-81")
        .expect("bounded exact-title request");
    let lexical = ChannelHit::new(exact.chunks[0].chunk_id.clone(), 0.01, 1).unwrap();
    let exact_chunk_id = exact.chunks[0].chunk_id.clone();
    assert!(
        try_batch(
            &catalog,
            &request,
            SearchChannel::Lexical,
            vec![lexical.clone()],
            [exact_chunk_id],
        )
        .is_err(),
        "a public title assertion cannot become rank authority before a real lexical adapter exists"
    );
    let hits = catalog
        .rank_note_hits(
            batch(
                &catalog,
                &request,
                SearchChannel::Lexical,
                vec![lexical],
                [],
            ),
            Some(batch(
                &catalog,
                &request,
                SearchChannel::Semantic,
                vec![ChannelHit::new(related.chunks[0].chunk_id.clone(), 1.0, 1).unwrap()],
                [],
            )),
            request,
        )
        .expect("ranked hits");

    assert_eq!(
        hits.first().map(|hit| hit.note_id.as_str()),
        Some("a-related"),
        "without trusted same-query title-field evidence, deterministic tie-breaking applies"
    );
    assert!(
        hits.iter()
            .all(|hit| !hit.rank_evidence.exact_lexical_title)
    );
}

#[test]
fn duplicate_chunks_collapse_to_one_note_without_losing_the_best_matching_passage() {
    let projection = chunk_note(
        &note(
            "vault-a",
            "n-duplicate",
            "Duplicated paragraphs",
            "The same evidence appears here.\n\nThe same evidence appears here.",
        ),
        &ChunkingPolicy::default(),
    )
    .expect("chunks");
    let mut catalog = ChunkCatalog::new("vault-a").unwrap();
    publish_semantic(&mut catalog, &projection);
    let semantic = semantic_hits(
        &[1.0, 0.0],
        &projection
            .chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| {
                let vector = if chunk.kind == ChunkKind::Title {
                    vec![-1.0, 0.0]
                } else if index == 1 {
                    vec![1.0, 0.0]
                } else {
                    vec![0.9, 0.1]
                };
                (chunk.chunk_id.clone(), vector)
            })
            .collect::<Vec<_>>(),
    );
    let request = request(5, None);
    let hits = catalog
        .rank_note_hits(
            batch(&catalog, &request, SearchChannel::Lexical, vec![], []),
            Some(batch(
                &catalog,
                &request,
                SearchChannel::Semantic,
                semantic,
                [],
            )),
            request,
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].note_id, "n-duplicate");
    assert_eq!(hits[0].snippet, "The same evidence appears here.");
}

#[test]
fn stable_chunk_identity_reuses_unchanged_tail_but_not_changed_text_and_disambiguates_duplicates() {
    let before = chunk_note(
        &note(
            "vault-a",
            "page-1",
            "Project",
            "Alpha stable paragraph.\n\nUnchanged tail paragraph.",
        ),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    let after_insert = chunk_note(
        &note(
            "vault-a",
            "page-1",
            "Project",
            "New leading paragraph.\n\nAlpha stable paragraph.\n\nUnchanged tail paragraph.",
        ),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    let before_tail = before
        .chunks
        .iter()
        .find(|chunk| chunk.text == "Unchanged tail paragraph.")
        .unwrap();
    let after_tail = after_insert
        .chunks
        .iter()
        .find(|chunk| chunk.text == "Unchanged tail paragraph.")
        .unwrap();
    assert_eq!(before_tail.logical_id, after_tail.logical_id);
    assert_eq!(before_tail.content_digest, after_tail.content_digest);
    let before_title = before
        .chunks
        .iter()
        .find(|chunk| chunk.kind == ChunkKind::Title)
        .unwrap();
    let after_title = after_insert
        .chunks
        .iter()
        .find(|chunk| chunk.kind == ChunkKind::Title)
        .unwrap();
    assert_eq!(before_title.logical_id, after_title.logical_id);

    let after_edit = chunk_note(
        &note(
            "vault-a",
            "page-1",
            "Project",
            "Alpha changed paragraph.\n\nUnchanged tail paragraph.",
        ),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    let before_alpha = before
        .chunks
        .iter()
        .find(|chunk| chunk.text == "Alpha stable paragraph.")
        .unwrap();
    let after_alpha = after_edit
        .chunks
        .iter()
        .find(|chunk| chunk.text == "Alpha changed paragraph.")
        .unwrap();
    assert_ne!(before_alpha.logical_id, after_alpha.logical_id);
    assert_ne!(before_alpha.content_digest, after_alpha.content_digest);
    assert_ne!(
        before_alpha.note_revision_digest,
        after_alpha.note_revision_digest
    );
    assert!(
        !after_edit
            .chunks
            .iter()
            .any(|chunk| chunk.logical_id == before_alpha.logical_id)
    );

    let duplicates = chunk_note(
        &note(
            "vault-a",
            "page-2",
            "Duplicates",
            "Same text.\n\nSame text.",
        ),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    let duplicate_ids: Vec<_> = duplicates
        .chunks
        .iter()
        .filter(|chunk| chunk.text == "Same text.")
        .map(|chunk| chunk.logical_id.as_str())
        .collect();
    assert_eq!(duplicate_ids.len(), 2);
    assert_ne!(duplicate_ids[0], duplicate_ids[1]);

    let duplicate_insert = chunk_note(
        &note(
            "vault-a",
            "page-2",
            "Duplicates",
            "Same text.\n\nSame text.\n\nSame text.",
        ),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    let duplicate_insert_ids: Vec<_> = duplicate_insert
        .chunks
        .iter()
        .filter(|chunk| chunk.text == "Same text.")
        .map(|chunk| chunk.logical_id.as_str())
        .collect();
    assert!(
        duplicate_ids
            .iter()
            .all(|before_id| !duplicate_insert_ids.contains(before_id))
    );

    let headings_before = chunk_note(
        &note(
            "vault-a",
            "page-3",
            "Reorder",
            "# One\n\nFirst paragraph.\n\n# Two\n\nSecond paragraph.",
        ),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    let headings_after = chunk_note(
        &note(
            "vault-a",
            "page-3",
            "Reorder",
            "# Two\n\nSecond paragraph.\n\n# One\n\nFirst paragraph.",
        ),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    for text in ["# One", "# Two", "First paragraph.", "Second paragraph."] {
        let before_chunk = headings_before
            .chunks
            .iter()
            .find(|chunk| chunk.text == text)
            .unwrap();
        let after_chunk = headings_after
            .chunks
            .iter()
            .find(|chunk| chunk.text == text)
            .unwrap();
        assert_eq!(before_chunk.logical_id, after_chunk.logical_id);
    }
}

#[test]
fn projections_reject_forged_identity_revision_title_and_body_range_receipts() {
    let projection = chunk_note(
        &note(
            "vault-a",
            "receipt-page",
            "Receipt",
            "First paragraph.\n\nSecond paragraph.",
        ),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    let body_index = projection
        .chunks
        .iter()
        .position(|chunk| chunk.text == "First paragraph.")
        .unwrap();
    let second_range = projection
        .chunks
        .iter()
        .find(|chunk| chunk.text == "Second paragraph.")
        .unwrap()
        .body_range
        .clone()
        .unwrap();

    let mut forged_logical_id = projection.clone();
    forged_logical_id.chunks[body_index].logical_id =
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into();
    let mut forged_chunk_id = projection.clone();
    forged_chunk_id.chunks[body_index].chunk_id = format!(
        "pc{CHUNK_FORMAT_VERSION}:sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    );
    let mut mixed_revision = projection.clone();
    mixed_revision.chunks[body_index].note_revision_digest =
        "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".into();
    let mut forged_title = projection.clone();
    forged_title.chunks[body_index].title = "Other title".into();
    let mut swapped_range = projection;
    swapped_range.chunks[body_index].body_range = Some(second_range);

    for forged in [
        forged_logical_id,
        forged_chunk_id,
        mixed_revision,
        forged_title,
        swapped_range,
    ] {
        let mut catalog = ChunkCatalog::new("vault-a").unwrap();
        assert!(
            catalog
                .publish(catalog.publication_token(), forged, manifest("vault-a", 1),)
                .is_err()
        );
        assert_eq!(catalog.generation(), 0);
    }
}

#[test]
fn note_and_query_identities_reject_control_and_path_like_values() {
    for (vault_id, note_id) in [
        ("vault\0", "page"),
        ("vault", "page\nnext"),
        ("vault", "../page"),
        ("vault", "/absolute-page"),
    ] {
        assert!(NoteInput::new(vault_id, note_id, "Title", "Body").is_err());
    }

    for (vault_id, limit, origin_note_id) in [
        ("vault\0", 1, None),
        ("vault-a", 0, None),
        ("vault-a", 65, None),
        ("vault-a", 1, Some("../origin")),
    ] {
        assert!(
            HybridRequest::new(
                vault_id,
                "natural paragraph query",
                limit,
                origin_note_id.map(str::to_string),
            )
            .is_err()
        );
    }
    assert!(request(1, None).with_exact_title("title\nnext").is_err());
    assert!(
        request(1, None)
            .with_exact_title(&"x".repeat(2_049))
            .is_err()
    );
}

#[test]
fn exact_query_bytes_and_input_policies_bind_each_request_and_lease() {
    let first = request_with_query("first paragraph query", 1, None);
    let second = request_with_query("second paragraph query", 1, None);
    let composed = request_with_query("caf\u{00e9}", 1, None);
    let decomposed = request_with_query("cafe\u{0301}", 1, None);

    assert_ne!(first.digest(), second.digest());
    assert_ne!(composed.digest(), decomposed.digest());
    for query in ["", "   ", "contains\ncontrol", &"x".repeat(16 * 1024 + 1)] {
        assert!(HybridRequest::new("vault-a", query, 1, None).is_err());
    }

    let projection = chunk_note(
        &note("vault-a", "page", "Title", "Matching paragraph."),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    let mut catalog = ChunkCatalog::new("vault-a").unwrap();
    publish(&mut catalog, &projection);
    let first_lease = catalog.issue_search_lease(&first, 1).unwrap();
    let second_lease = catalog.issue_search_lease(&second, 1).unwrap();
    assert_ne!(
        first_lease, second_lease,
        "same vault envelope with different exact query bytes must not share a lease identity"
    );
    let hit = ChannelHit::new(projection.chunks[0].chunk_id.clone(), 1.0, 1).unwrap();
    assert!(
        catalog
            .rank_note_hits(
                batch(&catalog, &first, SearchChannel::Lexical, vec![hit], []),
                None,
                second,
            )
            .is_err(),
        "a structurally bound assertion for one query may not rank under another query envelope"
    );
}

#[test]
fn chunk_projection_reports_exact_utf8_ranges_kinds_and_partial_coverage() {
    let projection = chunk_note(
        &note(
            "vault-a",
            "unicode-page",
            "Title 🇦🇶",
            "# Heading\r\n\r\nemoji 🇦🇶 and combining e\u{301}.\r\n\r\n```swift\r\nlet planet = \"土星\"\r\n```",
        ),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    let body = "# Heading\r\n\r\nemoji 🇦🇶 and combining e\u{301}.\r\n\r\n```swift\r\nlet planet = \"土星\"\r\n```";
    assert!(
        projection
            .chunks
            .iter()
            .any(|chunk| chunk.kind == ChunkKind::Title && chunk.body_range.is_none())
    );
    assert!(
        projection
            .chunks
            .iter()
            .any(|chunk| chunk.kind == ChunkKind::Heading)
    );
    assert!(
        projection
            .chunks
            .iter()
            .any(|chunk| chunk.kind == ChunkKind::Code)
    );
    for chunk in projection
        .chunks
        .iter()
        .filter(|chunk| chunk.body_range.is_some())
    {
        let range = chunk.body_range.as_ref().unwrap();
        assert!(body.is_char_boundary(range.start_byte));
        assert!(body.is_char_boundary(range.end_byte));
        assert_eq!(&body[range.start_byte..range.end_byte], chunk.text);
    }

    let partial = chunk_note(
        &note("vault-a", "bounded", "Bounded", &"word ".repeat(12_000)),
        &ChunkingPolicy {
            max_chunk_bytes: 128,
            max_chunks_per_note: 2,
            overlap_bytes: 16,
        },
    )
    .unwrap();
    assert!(matches!(
        partial.coverage.status,
        ProjectionStatus::Partial { .. }
    ));
    assert!(partial.coverage.indexed_body_bytes < partial.coverage.total_body_bytes);
    let mut catalog = ChunkCatalog::new("vault-a").unwrap();
    publish(&mut catalog, &partial);
    assert_eq!(catalog.generation(), 1);
}

#[test]
fn invalid_vectors_and_scores_fail_closed_without_nonfinite_rank_output() {
    assert!(cosine_search(&[0.0, 0.0], &[], 1).is_err());
    assert!(cosine_search(&[f32::NAN], &[], 1).is_err());
    assert!(cosine_search(&[f32::NEG_INFINITY], &[], 1).is_err());
    assert!(
        cosine_search(
            &[1.0, 0.0],
            &[VectorFixture {
                chunk_id: "bad".into(),
                vector: vec![f32::INFINITY, 0.0],
            }],
            1,
        )
        .is_err()
    );
    assert!(
        cosine_search(
            &[1.0, 0.0],
            &[VectorFixture {
                chunk_id: "wrong-dimension".into(),
                vector: vec![1.0],
            }],
            1,
        )
        .is_err()
    );
    assert!(ChannelHit::new("bad-score".into(), f32::NEG_INFINITY, 1).is_err());
    assert!(ChannelHit::new("nan-score".into(), f32::NAN, 1).is_err());
    assert!(ChannelHit::new("bad-rank".into(), 0.0, 0).is_err());
    assert!(ChannelHit::new("high-rank".into(), 0.0, 257).is_err());
    assert!(cosine_search(&[1.0e-10], &[], 1).is_err());
    assert!(cosine_search(&[f32::MAX], &[], 1).is_ok());
    assert!(cosine_search(&vec![1.0; 4_097], &[], 1).is_err());
    let orthogonal = cosine_search(
        &[1.0, 0.0],
        &[VectorFixture {
            chunk_id: "pc2:sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                .into(),
            vector: vec![0.0, 1.0],
        }],
        1,
    )
    .unwrap();
    assert_eq!(orthogonal[0].raw_score, 0.0);
    let too_many_vectors = vec![
        VectorFixture {
            chunk_id: "not-validated-after-count-cap".into(),
            vector: vec![1.0],
        };
        16_385
    ];
    assert!(cosine_search(&[1.0], &too_many_vectors, 1).is_err());
}

#[test]
fn manifest_vault_dimension_and_generation_mismatches_leave_catalog_unchanged() {
    let projection = chunk_note(
        &note("vault-a", "page", "Title", "Body"),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    let mut catalog = ChunkCatalog::new("vault-a").unwrap();
    let token = catalog.publication_token();
    let wrong_vault = manifest("vault-b", 1);
    assert!(
        catalog
            .publish(token, projection.clone(), wrong_vault)
            .is_err()
    );
    assert_eq!(catalog.generation(), 0);
    assert_eq!(catalog.chunk_count(), 0);

    let token = catalog.publication_token();
    let mut wrong_dimension = manifest("vault-a", 1);
    wrong_dimension.dimension = 0;
    assert!(
        catalog
            .publish(token, projection.clone(), wrong_dimension)
            .is_err()
    );
    assert_eq!(catalog.generation(), 0);

    for partial_vector_contract in [
        {
            let mut manifest = manifest("vault-a", 1);
            manifest.model_descriptor_digest = Some(
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
            );
            manifest
        },
        {
            let mut manifest = manifest("vault-a", 1);
            manifest.dimension = Some(256);
            manifest
        },
        {
            let mut manifest = manifest("vault-a", 1);
            manifest.vector_generation = Some(1);
            manifest
        },
        {
            let mut manifest = manifest("vault-a", 1);
            manifest.normalization = Some(VectorNormalization::InjectedCosineL2);
            manifest
        },
        {
            let mut manifest = manifest("vault-a", 1);
            manifest.vector_receipt = Some(ChannelReceipt::Pending {
                artifact_digest:
                    "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".into(),
            });
            manifest
        },
        {
            let mut manifest = semantic_manifest("vault-a", 1);
            manifest.dimension = Some(0);
            manifest
        },
    ] {
        assert!(
            catalog
                .publish(
                    catalog.publication_token(),
                    projection.clone(),
                    partial_vector_contract,
                )
                .is_err()
        );
        assert_eq!(catalog.generation(), 0);
    }

    for invalid_manifest in [
        {
            let mut manifest = manifest("vault-a", 1);
            manifest.schema_version = 0;
            manifest
        },
        {
            let mut manifest = manifest("vault-a", 1);
            manifest.chunk_format_version = CHUNK_FORMAT_VERSION + 1;
            manifest
        },
        {
            let mut manifest = manifest("vault-a", 1);
            manifest.model_descriptor_digest = Some("sha256:not-a-real-digest".into());
            manifest
        },
        {
            let mut manifest = manifest("vault-a", 1);
            manifest.chunk_map_digest = Some(
                "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".into(),
            );
            manifest
        },
        {
            let mut manifest = manifest("vault-a", 1);
            manifest.rank_policy_version = 0;
            manifest
        },
        {
            let mut manifest = manifest("vault-a", 1);
            manifest.lexical_generation = 0;
            manifest
        },
    ] {
        assert!(
            catalog
                .publish(
                    catalog.publication_token(),
                    projection.clone(),
                    invalid_manifest
                )
                .is_err()
        );
        assert_eq!(catalog.generation(), 0);
    }

    let token = catalog.publication_token();
    assert!(
        catalog
            .publish(token, projection, manifest("vault-a", 2))
            .is_err()
    );
    assert_eq!(catalog.generation(), 0);

    assert!(
        GenerationManifest::new(
            "vault-a",
            1,
            Some("sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into()),
            Some(256),
            CHUNK_FORMAT_VERSION,
            1,
            Some(1),
            policy(),
            Some(VectorNormalization::InjectedCosineL2),
            SemanticAvailability::Available,
            "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            None,
        )
        .is_err()
    );
}

#[test]
fn cancelled_and_stale_publications_cannot_replace_or_delete_current_chunks() {
    let first = chunk_note(
        &note("vault-a", "page", "Title", "First revision."),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    let second = chunk_note(
        &note("vault-a", "page", "Title", "Second revision."),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    let mut catalog = ChunkCatalog::new("vault-a").unwrap();

    let cancelled = catalog.publication_token().cancelled();
    assert!(
        catalog
            .publish(cancelled, first.clone(), manifest("vault-a", 1))
            .is_err()
    );
    assert_eq!(catalog.generation(), 0);

    let stale = catalog.publication_token();
    publish(&mut catalog, &first);
    let published_manifest = catalog.manifest().cloned().unwrap();
    assert!(published_manifest.chunking_policy_digest.is_some());
    assert!(published_manifest.chunk_map_digest.is_some());
    assert!(published_manifest.note_set_digest.is_some());
    assert!(matches!(
        published_manifest.lexical_receipt,
        ChannelReceipt::Bound { .. }
    ));
    assert_eq!(published_manifest.vector_generation, None);
    assert_eq!(published_manifest.vector_receipt, None);
    assert!(
        catalog
            .publish(stale, second.clone(), manifest("vault-a", 1))
            .is_err()
    );
    assert_eq!(catalog.generation(), 1);
    assert!(
        catalog
            .chunks_for_note("page")
            .iter()
            .any(|chunk| chunk.text == "First revision.")
    );
    assert!(
        catalog
            .chunks_for_note("page")
            .iter()
            .all(|chunk| chunk.text != "Second revision.")
    );

    let before_cancelled_delete = catalog.clone();
    assert!(
        catalog
            .remove_note(
                catalog.publication_token().cancelled(),
                "page",
                lexical_staging(),
            )
            .is_err()
    );
    assert_eq!(catalog, before_cancelled_delete);

    let stale_delete_token = catalog.publication_token();
    let delete_token = catalog.publication_token();
    let delete_receipt = catalog
        .remove_note(delete_token, "page", lexical_staging())
        .unwrap();
    assert_eq!(catalog.generation(), 2);
    assert!(catalog.chunks_for_note("page").is_empty());
    assert_eq!(
        delete_receipt.semantic_availability,
        SemanticAvailability::NoCandidateSelected
    );
    assert!(matches!(
        delete_receipt.lexical_receipt,
        ChannelReceipt::Bound { .. }
    ));
    assert!(delete_receipt.vector_receipt.is_none());
    assert!(catalog.manifest().is_some());
    assert!(
        catalog
            .reset(stale_delete_token, lexical_staging())
            .is_err()
    );
    let before_cancelled_reset = catalog.clone();
    assert!(
        catalog
            .reset(catalog.publication_token().cancelled(), lexical_staging(),)
            .is_err()
    );
    assert_eq!(catalog, before_cancelled_reset);
    let reset_receipt = catalog
        .reset(catalog.publication_token(), lexical_staging())
        .unwrap();
    assert_eq!(catalog.generation(), 3);
    assert!(matches!(
        reset_receipt.lexical_receipt,
        ChannelReceipt::Bound { .. }
    ));
    assert!(reset_receipt.vector_receipt.is_none());
}

#[test]
fn vault_isolation_and_missing_semantic_assets_preserve_only_local_lexical_results() {
    let projection_a = chunk_note(
        &note("vault-a", "same-page", "Rare title", "A body."),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    let projection_b = chunk_note(
        &note("vault-b", "same-page", "Rare title", "A body."),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    let mut catalog = ChunkCatalog::new("vault-a").unwrap();
    publish(&mut catalog, &projection_a);
    assert!(
        catalog
            .publish(
                catalog.publication_token(),
                projection_b,
                manifest("vault-a", 2)
            )
            .is_err()
    );
    assert_eq!(catalog.chunk_count(), projection_a.chunks.len());

    let request = request(5, None);
    let hits = catalog
        .rank_note_hits(
            batch(
                &catalog,
                &request,
                SearchChannel::Lexical,
                vec![ChannelHit::new(projection_a.chunks[0].chunk_id.clone(), 1.0, 1).unwrap()],
                [],
            ),
            None,
            request,
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].source, "lexical-fallback");
    assert!(hits[0].rank_evidence.semantic.is_none());
}

#[test]
fn origin_note_is_excluded_before_final_note_ranking() {
    let origin = chunk_note(
        &note("vault-a", "origin-page", "Origin", "The current note body."),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    let related = chunk_note(
        &note(
            "vault-a",
            "related-page",
            "Related",
            "A matching external note.",
        ),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    let mut catalog = ChunkCatalog::new("vault-a").unwrap();
    publish(&mut catalog, &origin);
    publish(&mut catalog, &related);

    let request = request(5, Some("origin-page"));
    assert!(
        try_batch(
            &catalog,
            &request,
            SearchChannel::Lexical,
            vec![
                ChannelHit::new(origin.chunks[0].chunk_id.clone(), 100.0, 1).unwrap(),
                ChannelHit::new(related.chunks[0].chunk_id.clone(), 1.0, 2).unwrap(),
            ],
            [],
        )
        .is_err(),
        "origin rows must be removed before a bounded channel assertion"
    );
    let hits = catalog
        .rank_note_hits(
            batch(
                &catalog,
                &request,
                SearchChannel::Lexical,
                vec![ChannelHit::new(related.chunks[0].chunk_id.clone(), 1.0, 1).unwrap()],
                [],
            ),
            None,
            request,
        )
        .unwrap();

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].note_id, "related-page");
}

#[test]
fn forged_projection_coverage_cannot_publish() {
    let projection = chunk_note(
        &note(
            "vault-a",
            "coverage-page",
            "Coverage",
            "First paragraph.\n\nSecond paragraph.",
        ),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    let mut dishonest_count = projection.clone();
    dishonest_count.coverage.indexed_body_bytes = 0;
    let mut catalog = ChunkCatalog::new("vault-a").unwrap();
    assert!(
        catalog
            .publish(
                catalog.publication_token(),
                dishonest_count,
                manifest("vault-a", 1),
            )
            .is_err()
    );
    assert_eq!(catalog.generation(), 0);

    let mut dishonest_partial = projection;
    dishonest_partial.coverage.status = ProjectionStatus::Partial {
        omitted_from_body_byte: dishonest_partial.body_byte_len,
        reason: ProjectionPartialReason::ChunkLimit,
    };
    assert!(
        catalog
            .publish(
                catalog.publication_token(),
                dishonest_partial,
                manifest("vault-a", 1),
            )
            .is_err()
    );
    assert_eq!(catalog.generation(), 0);
}

#[test]
fn complete_projection_must_equal_the_canonical_projector_output() {
    let projection = chunk_note(
        &note(
            "vault-a",
            "canonical-page",
            "Canonical",
            "# First\n\nFirst paragraph.\n\n# Second\n\nSecond paragraph.",
        ),
        &ChunkingPolicy::default(),
    )
    .unwrap();

    let mut missing = projection.clone();
    missing
        .chunks
        .retain(|chunk| chunk.text != "Second paragraph.");
    let mut reordered = projection.clone();
    reordered.chunks.reverse();
    let mut extra = projection.clone();
    extra.chunks.push(projection.chunks[0].clone());

    let partial = chunk_note(
        &note(
            "vault-a",
            "partial-page",
            "Partial",
            &"word ".repeat(12_000),
        ),
        &ChunkingPolicy {
            max_chunk_bytes: 128,
            max_chunks_per_note: 2,
            overlap_bytes: 16,
        },
    )
    .unwrap();
    let mut forged_complete = partial;
    forged_complete.coverage.status = ProjectionStatus::Complete;

    for forged in [missing, reordered, extra, forged_complete] {
        let mut catalog = ChunkCatalog::new("vault-a").unwrap();
        assert!(
            catalog
                .publish(catalog.publication_token(), forged, manifest("vault-a", 1))
                .is_err()
        );
        assert_eq!(catalog.generation(), 0);
        assert_eq!(catalog.chunk_count(), 0);
    }
}

#[test]
fn catalog_receipt_is_the_only_semantic_authority_and_mutations_keep_receipts_honest() {
    let projection = chunk_note(
        &note("vault-a", "page", "Title", "Matching paragraph."),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    let mut catalog = ChunkCatalog::new("vault-a").unwrap();
    publish(&mut catalog, &projection);
    let chunk_id = projection.chunks[0].chunk_id.clone();
    let lexical = ChannelHit::new(chunk_id.clone(), 0.5, 1).unwrap();
    let _semantic = ChannelHit::new(chunk_id, 1.0, 1).unwrap();

    let request = request(5, None);
    let lexical_only = catalog
        .rank_note_hits(
            batch(
                &catalog,
                &request,
                SearchChannel::Lexical,
                vec![lexical],
                [],
            ),
            None,
            request,
        )
        .unwrap();
    assert_eq!(lexical_only.len(), 1);
    assert_eq!(lexical_only[0].source, "lexical-fallback");
    assert!(lexical_only[0].rank_evidence.semantic.is_none());

    let published = catalog.manifest().cloned().unwrap();
    let base_digest = published.digest();
    for altered in [
        {
            let mut manifest = published.clone();
            manifest.rank_policy_version += 1;
            manifest
        },
        {
            let mut manifest = published.clone();
            manifest.rank_policy_digest =
                "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".into();
            manifest
        },
        {
            let mut manifest = published.clone();
            manifest.normalization = Some(VectorNormalization::InjectedCosineL2);
            manifest.semantic_availability = SemanticAvailability::Rebuilding;
            manifest
        },
        {
            let mut manifest = published.clone();
            manifest.lexical_receipt = ChannelReceipt::Bound {
                artifact_digest:
                    "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into(),
                staged_state_digest:
                    "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".into(),
            };
            manifest
        },
    ] {
        assert_ne!(base_digest, altered.digest());
    }

    assert!(LexicalStagingReceipt::new("not-a-digest").is_err());
    let source = include_str!("../src/backend/free_semantic.rs");
    assert!(source.contains("lexical_staging: LexicalStagingReceipt"));
    let mutation_start = source.find("fn stage_degraded_mutation_receipt").unwrap();
    let mutation_end = source[mutation_start..]
        .find("fn validate_token")
        .map(|offset| mutation_start + offset)
        .unwrap();
    assert!(!source[mutation_start..mutation_end].contains("GenerationManifest,"));
    assert!(
        catalog
            .remove_note(
                catalog.publication_token().cancelled(),
                "page",
                lexical_staging(),
            )
            .is_err()
    );
    assert_eq!(catalog.generation(), 1);
    assert_eq!(catalog.chunk_count(), projection.chunks.len());
}

#[test]
fn vault_bound_request_and_duplicate_channel_entries_fail_before_candidate_ranking() {
    let projection = chunk_note(
        &note("vault-a", "page", "Title", "Matching paragraph."),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    let mut catalog = ChunkCatalog::new("vault-a").unwrap();
    publish(&mut catalog, &projection);
    let chunk_id = projection.chunks[0].chunk_id.clone();
    let rank_one = ChannelHit::new(chunk_id.clone(), 1.0, 1).unwrap();
    let rank_two = ChannelHit::new(chunk_id, 0.5, 2).unwrap();
    let same_rank_different_chunk =
        ChannelHit::new(projection.chunks[1].chunk_id.clone(), 0.25, 1).unwrap();

    let request = request(1, None);
    let wrong_request = HybridRequest::new("vault-b", "natural paragraph query", 1, None).unwrap();
    assert!(
        try_batch(
            &catalog,
            &wrong_request,
            SearchChannel::Lexical,
            vec![rank_one.clone()],
            [],
        )
        .is_err()
    );
    assert!(
        try_batch(
            &catalog,
            &request,
            SearchChannel::Lexical,
            vec![rank_one.clone(), same_rank_different_chunk],
            [],
        )
        .is_err()
    );
    assert!(
        try_batch(
            &catalog,
            &request,
            SearchChannel::Lexical,
            vec![rank_one.clone(), rank_two],
            [],
        )
        .is_err()
    );
    assert!(
        catalog
            .rank_note_hits(
                batch(
                    &catalog,
                    &request,
                    SearchChannel::Lexical,
                    vec![rank_one.clone()],
                    [],
                ),
                None,
                request.clone(),
            )
            .is_ok()
    );
    let noncontiguous_rank =
        ChannelHit::new(projection.chunks[1].chunk_id.clone(), 0.25, 3).unwrap();
    assert!(
        try_batch(
            &catalog,
            &request(2, None),
            SearchChannel::Lexical,
            vec![
                ChannelHit::new(projection.chunks[0].chunk_id.clone(), 1.0, 1).unwrap(),
                noncontiguous_rank
            ],
            [],
        )
        .is_err()
    );
    assert!(
        catalog
            .rank_note_hits(
                batch(
                    &catalog,
                    &request,
                    SearchChannel::Lexical,
                    vec![ChannelHit::new(projection.chunks[0].chunk_id.clone(), 1.0, 1).unwrap()],
                    [],
                ),
                None,
                request,
            )
            .is_ok()
    );
}

#[test]
fn semantic_mutations_clear_vector_contracts_and_policy_migrations_require_full_rebuild() {
    let initial_policy = ChunkingPolicy::default();
    let replacement_policy = ChunkingPolicy {
        max_chunk_bytes: 1_024,
        max_chunks_per_note: 64,
        overlap_bytes: 64,
    };
    let initial = chunk_note(
        &note(
            "vault-a",
            "page",
            "Title",
            "A paragraph retained across the migration.",
        ),
        &initial_policy,
    )
    .unwrap();
    let replacement = chunk_note(
        &note(
            "vault-a",
            "page",
            "Title",
            "A paragraph retained across the migration.",
        ),
        &replacement_policy,
    )
    .unwrap();
    let mut catalog = ChunkCatalog::new("vault-a").unwrap();
    publish_semantic(&mut catalog, &initial);
    let before_rejected_incremental = catalog.clone();
    assert!(
        catalog
            .publish(
                catalog.publication_token(),
                replacement.clone(),
                semantic_manifest("vault-a", 2),
            )
            .is_err(),
        "an incremental publication may not relabel retained chunks with a new policy"
    );
    assert_eq!(catalog, before_rejected_incremental);

    let delete = catalog
        .remove_note(catalog.publication_token(), "page", lexical_staging())
        .unwrap();
    assert_eq!(
        delete.semantic_availability,
        SemanticAvailability::NoCandidateSelected
    );
    assert_eq!(delete.model_descriptor_digest, None);
    assert_eq!(delete.dimension, None);
    assert_eq!(delete.normalization, None);
    assert_eq!(delete.vector_generation, None);
    assert_eq!(delete.vector_receipt, None);

    let rebuild = catalog
        .rebuild(
            catalog.publication_token(),
            replacement_policy,
            vec![replacement],
            semantic_manifest("vault-a", 3),
        )
        .unwrap();
    assert_eq!(catalog.generation(), 3);
    assert_eq!(
        rebuild.semantic_availability,
        SemanticAvailability::Available
    );
    assert!(rebuild.vector_receipt.is_some());
    assert!(catalog.chunks_for_note("page").len() >= 1);
}

#[test]
fn channel_batches_require_a_catalog_lease_and_disclose_untrusted_assertion_scope() {
    let source = include_str!("../src/backend/free_semantic.rs");
    assert!(source.contains("pub struct SearchLease"));
    assert!(source.contains("lease: SearchLease,"));
    assert!(!source.contains("#[derive(Clone, Debug, PartialEq, Eq)]\npub struct SearchLease"));
    assert!(source.contains("complete_untrusted_channel_assertion"));
    assert!(source.contains("not* replay-resistant evidence that lexical or vector retrieval ran"));
    assert!(source.contains("Its counts and completion label are caller assertions"));
    assert!(!source.contains("exact_title_chunk_ids"));
    assert!(
        !source.contains(
            "impl ChannelBatch {\n    #[allow(clippy::too_many_arguments)]\n    pub fn new"
        )
    );

    let projection = chunk_note(
        &note("vault-a", "page", "Title", "Matching paragraph."),
        &ChunkingPolicy::default(),
    )
    .unwrap();
    let mut catalog = ChunkCatalog::new("vault-a").unwrap();
    publish(&mut catalog, &projection);
    let request = request(1, None);
    let lease = catalog.issue_search_lease(&request, 1).unwrap();
    let hit = ChannelHit::new(projection.chunks[0].chunk_id.clone(), 1.0, 1).unwrap();
    assert!(
        catalog
            .complete_untrusted_channel_assertion(
                lease,
                SearchChannel::Lexical,
                ChannelCompletion::Complete,
                0,
                0,
                vec![hit.clone()],
                BTreeSet::new(),
            )
            .is_err(),
        "honest complete accounting cannot claim zero eligible rows while returning a hit"
    );
    assert!(
        catalog
            .complete_untrusted_channel_assertion(
                catalog.issue_search_lease(&request, 1).unwrap(),
                SearchChannel::Lexical,
                ChannelCompletion::Complete,
                1,
                0,
                vec![hit],
                BTreeSet::new(),
            )
            .is_ok()
    );
}

#[test]
fn content_bearing_contract_debug_is_bounded_and_redacts_plaintext_and_digests() {
    let note = NoteInput::new(
        "vault-canary",
        "note-canary",
        "title-canary",
        "body-canary unique paragraph",
    )
    .unwrap();
    let projection = chunk_note(&note, &ChunkingPolicy::default()).unwrap();
    let request = HybridRequest::new("vault-canary", "query-canary", 1, None)
        .unwrap()
        .with_exact_title("exact-title-canary")
        .unwrap();
    let mut catalog = ChunkCatalog::new("vault-canary").unwrap();
    publish(&mut catalog, &projection);
    let lease = catalog.issue_search_lease(&request, 1).unwrap();
    let ranked = catalog
        .rank_note_hits(
            batch(
                &catalog,
                &request,
                SearchChannel::Lexical,
                vec![ChannelHit::new(projection.chunks[0].chunk_id.clone(), 1.0, 1).unwrap()],
                [],
            ),
            None,
            request.clone(),
        )
        .unwrap();

    let diagnostics = [
        format!("{note:?}"),
        format!("{:?}", projection.chunks.first().unwrap()),
        format!("{projection:?}"),
        format!("{request:?}"),
        format!("{lease:?}"),
        format!("{catalog:?}"),
        format!("{:?}", ranked.first().unwrap()),
    ];
    let request_digest = request.digest();
    for diagnostic in diagnostics {
        assert!(
            diagnostic.len() < 1_024,
            "redacted diagnostics stay bounded"
        );
        for canary in [
            "vault-canary",
            "note-canary",
            "title-canary",
            "body-canary",
            "query-canary",
            "exact-title-canary",
            request_digest.as_str(),
        ] {
            assert!(
                !diagnostic.contains(canary),
                "diagnostic leaked sensitive content"
            );
        }
    }

    let source = include_str!("../src/backend/free_semantic.rs");
    for type_name in [
        "NoteInput",
        "ParagraphChunk",
        "ChunkProjection",
        "HybridRequest",
        "SearchLease",
        "RankedNoteHit",
        "ChunkCatalog",
    ] {
        assert!(
            source.contains(&format!("impl fmt::Debug for {type_name}")),
            "content-bearing contract {type_name} requires an explicit redacted Debug implementation"
        );
    }
    assert!(!source.contains("#[derive(Clone, Debug, PartialEq, Eq)]\npub struct NoteInput"));
    assert!(!source.contains("#[derive(Debug, PartialEq, Eq)]\npub struct SearchLease"));
    assert!(!source.contains("#[derive(Clone, Debug, PartialEq)]\npub struct RankedNoteHit"));
}
