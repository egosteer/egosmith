#!/bin/bash
# Materialize the patched chumpy package used for MANO pickle loading.
#
# The upstream source is kept as a clean git submodule at
# thirdparty/chumpy_upstream. This script copies that source into the ignored
# install path thirdparty/chumpy and applies the small setup.py patch needed for
# modern pip build isolation. The copy is disposable and can be regenerated.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SRC="thirdparty/chumpy_upstream"
DST="thirdparty/chumpy"
PATCH_FILE="patches/chumpy/setup.py.patch"

if [ ! -f "$SRC/setup.py" ]; then
    echo "[fetch_chumpy] populating submodule: $SRC"
    git submodule update --init "$SRC"
fi

if [ ! -f "$SRC/setup.py" ]; then
    echo "[fetch_chumpy] missing upstream source: $SRC/setup.py" >&2
    echo "  run: git submodule update --init $SRC" >&2
    exit 1
fi
if [ ! -f "$PATCH_FILE" ]; then
    echo "[fetch_chumpy] missing patch: $PATCH_FILE" >&2
    exit 1
fi

rm -rf "$DST"
mkdir -p "$(dirname "$DST")"
cp -a "$SRC" "$DST"
rm -rf "$DST/.git"
patch -s -d "$DST" -p1 < "$PATCH_FILE"

echo "[fetch_chumpy] materialized patched chumpy at $DST from $SRC ($(git -C "$SRC" rev-parse --short HEAD))."
