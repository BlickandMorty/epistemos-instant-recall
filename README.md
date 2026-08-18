# Epistemos Instant Recall

Local-first search that can be used in two different ways without maintaining
two different indexes:

1. **Sidebar Search** — type a deliberate query and get fast title + body
   results.
2. **Ambient Instant Recall** — keep writing normally and receive a few related
   notes without opening search. The note being edited is excluded.

This is an executable extraction from my Epistemos note-taking app, not a UI
mockup. The repository retains the original Git history of the Contextual
Shadows engine, while `research/epistemos-core-instant-recall` preserves the
earlier binary-quantized retrieval experiments as research provenance.

## What is fused

```text
                         one local document catalog
                                    |
               +--------------------+--------------------+
               |                    |                    |
        title/prefix rank      Tantivy BM25       hashed-trigram vector
               +--------------------+--------------------+
                                    |
                           weighted RRF (k=60)
                                    |
                   +----------------+----------------+
                   |                                 |
             Sidebar Search                  Ambient Recall
                                                excludes origin
```

The default vector channel is deterministic character-trigram similarity. It
is useful for spelling variation and partial phrasing, but it is **not described
as a neural semantic model**. The preserved `semantic` feature contains the
original Model2Vec + HNSW path and is opt-in because it downloads a model and
has a heavier dependency closure.

All indexing and querying is local. Nothing is uploaded.

## Run it

Install [Rust](https://rustup.rs/), then:

```bash
cargo build --release
recall init --index ./my-index
recall index ~/Documents/Notes --index ./my-index
recall search "deterministic systems" --index ./my-index
recall recall "The paragraph currently being written about deterministic systems" --index ./my-index
recall stats --index ./my-index
```

Output is JSON so editors, desktop launchers, Swift/Kotlin/Electron apps, and
agent tools can consume the same engine. The Rust library exposes
`search_sidebar` and `recall_ambient` as separate APIs. A UI should debounce
sidebar keystrokes around 100 ms; ambient recall should run after an idle pause,
not on the rendering thread.

Supported inputs are plain-text notes, Markdown, HTML, TeX, common source-code
files, JSON/YAML/TOML/XML, SQL, and shell/PowerShell scripts. Files larger than
2 MiB, links, unreadable files, and binary formats are skipped. PDF/Office
content requires a caller-provided text extractor; the project does not pretend
to parse those formats as plain text.

## Desktop integration

| OS | Included | Boundary |
|---|---|---|
| Windows | Per-user Explorer context menu for folders; no admin required | Does not replace Explorer's native search box |
| macOS | CLI installation plus a Finder Quick Action template | Native Spotlight replacement requires a signed importer/extension |
| Linux | User-local CLI and `.desktop` launcher; Nautilus script | File-manager search-box behavior differs by desktop |

Every installer is reversible and supports a dry run. Run the matching script
in `integrations/`; see its inline help before installing.

## Verification

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo run --bin recall -- --index ./tmp-index index ./examples
cargo run --bin recall -- --index ./tmp-index search "note"
```

CI runs the format, lint, and test checks on Windows, macOS, and Linux. Search
results carry their contributing channels and measured query time so benchmark
claims can be reproduced instead of guessed.

## Lineage and licensing

- The root engine descends from `epistemos-shadow` (Tantivy/BM25, optional
  Model2Vec/HNSW, persistence, C ABI lineage).
- `research/epistemos-core-instant-recall` contains the earlier trigram,
  binary-quantization, Hamming retrieval, and fusion research.
- The portable facade and installers are the cross-platform extraction layer.

Copyright 2026 Jordan T. Conley. Licensed under Apache-2.0. This project is
experimental software: back up important files, inspect installer dry runs,
and do not treat ranking output as factual verification.
