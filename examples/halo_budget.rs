//! AMBIENT_RECALL_HALO_MASTER_PLAN §4 — minimal halo-budget harness.
//!
//! Runs a small fixture through `RealBackend::search()` and dumps the
//! per-stage `SearchTimings` as JSON to stdout. The CI gate in
//! `.github/workflows/ci.yml` runs this binary after the workspace
//! build to keep the per-stage signpost FFI wired (regression guard).
//!
//! Future expansion (post-V2.2 close-out): grow the fixture to N
//! search iterations + compute p50/p95/p99 + compare against the
//! doctrine ceilings (embed < 4 ms, ann < 10 ms, bm25 < 12 ms,
//! fusion < 5 ms, total < 40 ms). For now this is the smoke test that
//! ensures the SearchTimings struct + last_timings FFI exist + return
//! well-formed JSON.
//!
//! Skips with a non-fatal stderr message if the Model2Vec model isn't
//! cached locally (CI runners without ~/.cache/huggingface/ shouldn't
//! fail on the first cold pass).

use std::error::Error;

use epistemos_shadow::backend::RealBackend;
use epistemos_shadow::backend::ShadowBackend;
use epistemos_shadow::ShadowDocument;

fn main() -> Result<(), Box<dyn Error>> {
    let backend = match RealBackend::new() {
        Ok(b) => b,
        Err(e) => {
            // Soft-fail when Model2Vec download isn't possible. The
            // CI gate prints this and exits 0 so the binary's
            // existence + linkage is still verified.
            eprintln!(
                "halo_budget: skipping — RealBackend::new failed: {e:?} \
                 (likely no Model2Vec cache; full p99 bench needs a hot cache)"
            );
            println!("{{\"status\":\"skipped\",\"reason\":\"model2vec_unavailable\"}}");
            return Ok(());
        }
    };

    backend.insert_document(ShadowDocument {
        doc_id: "halo-budget-fixture".to_string(),
        domain: "note".to_string(),
        title: "Quarterly performance review".to_string(),
        body: "Revenue grew across regions; compute spend held flat; new \
               vault index latency dropped 40% versus the prior quarter."
            .to_string(),
        origin_vault_key: None,
    })?;

    let _hits = backend.search("revenue", "note", 5)?;
    let timings = backend.last_timings();

    let json = serde_json::to_string(&timings)?;
    println!("{json}");
    Ok(())
}
