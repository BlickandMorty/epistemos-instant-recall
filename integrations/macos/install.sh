#!/bin/sh
set -eu
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
BIN="$HOME/.local/bin/recall"
if [ "${1:-}" = "--uninstall" ]; then
  [ "$DRY_RUN" = 1 ] || rm -f "$BIN"
  echo "Removed the user-local CLI; index data was left intact."
  exit 0
fi
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
echo "Would build and install the CLI at $BIN."
echo "Use quick-action.sh as the Run Shell Script body in a Finder Quick Action."
[ "$DRY_RUN" = 1 ] && exit 0
cargo build --release --manifest-path "$ROOT/Cargo.toml"
mkdir -p "$(dirname "$BIN")"
cp "$ROOT/target/release/recall" "$BIN"
chmod +x "$BIN"
