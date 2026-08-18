#!/usr/bin/env sh
set -eu
DRY_RUN="${DRY_RUN:-0}"
PREFIX="${XDG_BIN_HOME:-$HOME/.local/bin}"
APPLICATIONS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
NAUTILUS="${XDG_DATA_HOME:-$HOME/.local/share}/nautilus/scripts"
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN=1; fi
if [ "${1:-}" = "--uninstall" ]; then
  [ "$DRY_RUN" = 1 ] || rm -f "$PREFIX/recall" "$APPLICATIONS/epistemos-recall.desktop" "$NAUTILUS/Index with Epistemos Recall"
  echo "Removed user-local launchers; index data was left intact."
  exit 0
fi
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
echo "Install recall in $PREFIX and desktop/file-manager launchers in user-local directories."
[ "$DRY_RUN" = 1 ] && exit 0
cargo build --release --manifest-path "$ROOT/Cargo.toml"
mkdir -p "$PREFIX" "$APPLICATIONS" "$NAUTILUS"
cp "$ROOT/target/release/recall" "$PREFIX/recall"
cp "$ROOT/integrations/linux/epistemos-recall.desktop" "$APPLICATIONS/"
cp "$ROOT/integrations/linux/nautilus-index.sh" "$NAUTILUS/Index with Epistemos Recall"
chmod +x "$PREFIX/recall" "$NAUTILUS/Index with Epistemos Recall"
