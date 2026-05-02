#!/bin/sh
# Lance le serveur ARIA sur macOS avec les libs Homebrew visibles par WeasyPrint (pango/gobject).
# Usage : ./run.sh [args uvicorn supplémentaires]
export DYLD_LIBRARY_PATH=/opt/homebrew/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
exec uv run uvicorn main:app --reload --port 8000 "$@"
