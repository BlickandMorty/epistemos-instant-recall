#!/usr/bin/env sh
set -eu
for path in ${NAUTILUS_SCRIPT_SELECTED_FILE_PATHS:-}; do recall index "$path"; done
