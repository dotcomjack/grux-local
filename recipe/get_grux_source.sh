#!/bin/sh
# Fetch the Grux source the v1 numbers were scored against and print the path to export as GRUX_SRC.
#   export GRUX_SRC="$(sh recipe/get_grux_source.sh)"
# Held-out v3 reads the tool catalogue at its own pinned tag (s2/holdout_v3.py, GRUX_TAG):
#   export GRUX_SRC_V3="$(GRUX_TAG=v3.3.1 sh recipe/get_grux_source.sh)"
set -e
TAG="${GRUX_TAG:-v3.1.0}"
DIR="${1:-${TMPDIR:-/tmp}/grux-$TAG}"
[ -d "$DIR/.git" ] || git -c advice.detachedHead=false clone -q --depth 1 --branch "$TAG" https://github.com/dotcomjack/grux.git "$DIR" 2>/dev/null
echo "$DIR/Grux-Mac/Sources/Grux"
