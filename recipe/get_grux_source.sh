#!/bin/sh
# Fetch the Grux source the v1 numbers were scored against and print the path to export as GRUX_SRC.
#   export GRUX_SRC="$(sh recipe/get_grux_source.sh)"
set -e
TAG=v3.1.0
DIR="${1:-${TMPDIR:-/tmp}/grux-$TAG}"
[ -d "$DIR/.git" ] || git -c advice.detachedHead=false clone -q --depth 1 --branch "$TAG" https://github.com/dotcomjack/grux.git "$DIR" 2>/dev/null
echo "$DIR/Grux-Mac/Sources/Grux"
