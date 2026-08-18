#!/bin/sh
set -eu
for path in "$@"; do "$HOME/.local/bin/recall" index "$path"; done
