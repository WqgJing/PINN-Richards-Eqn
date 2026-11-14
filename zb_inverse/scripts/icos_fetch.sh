#!/bin/bash
# Download ICOS Carbon Portal data objects of one station, each into its own new directory.
#
#   scripts/icos_fetch.sh <station> <object id> [<object id> ...]
#   e.g. scripts/icos_fetch.sh FR-Bil z3RB-bxCZ56yThkccIDCqxuM xFrwVCA1RxiQ4UyGinPgeqSZ
#
# The objects are under the ICOS CC BY 4.0 data licence (https://data.icos-cp.eu/licence). Requesting
# /licence_accept IS the acceptance of that licence ("Yes, I accept" on the portal): run this only for data
# you may use under it, and cite the objects (citation string in meta/<id>.json) in anything published.
# No account and no personal details are sent; the client name is generic.
#
# Layout (data/sites is git-ignored: raw station files are not committed):
#   data/sites/icos/<station>/<id>/raw/object.zip   as downloaded, sha256 checked against the open metadata
#   data/sites/icos/<station>/<id>/csv/             its files, unpacked without paths
#   output/site_search/icos/meta/<id>.json          the open metadata of the object
set -u
cd "$(dirname "$0")/.." || exit 1
UA="fem-validation-research-client/1.0"
station=$1; shift
case "$station" in *[!A-Za-z0-9-]*|"") echo "bad station name"; exit 2;; esac
meta=output/site_search/icos/meta
mkdir -p "$meta" || exit 1
rc=0
for id in "$@"; do
    case "$id" in *[!A-Za-z0-9_-]*|"") echo "bad object id: $id"; rc=2; continue;; esac
    d=data/sites/icos/$station/$id
    if [ -e "$d" ]; then echo "$station $id: $d exists, skipped"; continue; fi
    [ -s "$meta/$id.json" ] || curl -sS -m 60 -A "$UA" -H "Accept: application/json" -o "$meta/$id.json" \
        "https://meta.icos-cp.eu/objects/$id"
    want=$(jq -r '.hash // empty' "$meta/$id.json")
    name=$(jq -r '.fileName // empty' "$meta/$id.json")
    if [ -z "$want" ]; then echo "$station $id: no checksum in the metadata, not downloaded"; rc=1; continue; fi
    mkdir -p "$d/raw" || exit 1
    cj=$(mktemp)
    curl -sS -L -m 900 -A "$UA" -c "$cj" -b "$cj" -o "$d/raw/object.zip" \
        "https://data.icos-cp.eu/licence_accept?ids=%5B%22$id%22%5D"
    rm -f "$cj"
    got=$(sha256sum "$d/raw/object.zip" | cut -d' ' -f1 | xxd -r -p | base64 | tr '+/' '-_' | tr -d '=')
    if [ "$got" != "$want" ]; then echo "$station $id ($name): sha256 MISMATCH, not unpacked"; rc=1; continue; fi
    mkdir "$d/csv" && unzip -q -j "$d/raw/object.zip" -d "$d/csv" || { echo "$station $id: unzip failed"; rc=1; continue; }
    echo "$station $id ($name): sha256 OK, $(ls "$d/csv" | wc -l) file(s), $(du -sh "$d/csv" | cut -f1)"
done
exit $rc
