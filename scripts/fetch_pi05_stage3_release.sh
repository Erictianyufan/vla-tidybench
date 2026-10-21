#!/usr/bin/env bash
# Resume the seven large GitHub Release assets for the Stage-3 full checkpoint.
# Requires `gh auth login` or a temporary GH_TOKEN with access to the private repo.
set -euo pipefail

DEST="${1:?usage: fetch_pi05_stage3_release.sh DESTINATION_DIRECTORY}"
REPO="Erictianyufan/vla-tidybench-pi05-weights"
TAG="stage3-hard-recovery-2999-v1"
PREFIX="pi05-tidybench-stage3-hard-recovery-2999.tar.zst.part-"
mkdir -p "${DEST}"

mapfile -t ASSETS < <(gh api "repos/${REPO}/releases/tags/${TAG}" \
  --jq ".assets[] | select(.name | startswith(\"${PREFIX}\")) | [.name, (.id | tostring), (.size | tostring)] | @tsv")
if [[ "${#ASSETS[@]}" -ne 7 ]]; then
  echo "Expected seven Stage-3 checkpoint parts, found ${#ASSETS[@]}" >&2
  exit 2
fi

for ROW in "${ASSETS[@]}"; do
  IFS=$'\t' read -r NAME ID EXPECTED <<< "${ROW}"
  FILE="${DEST}/${NAME}"
  CURRENT=0
  [[ ! -f "${FILE}" ]] || CURRENT="$(stat -c %s "${FILE}")"
  if (( CURRENT > EXPECTED )); then
    echo "Oversized file; inspect before retrying: ${FILE}" >&2
    exit 2
  fi
  STALLED=0
  while (( CURRENT < EXPECTED )); do
    echo "${NAME}: resuming at ${CURRENT}/${EXPECTED} bytes"
    gh api -H 'Accept: application/octet-stream' \
      -H "Range: bytes=${CURRENT}-" \
      "repos/${REPO}/releases/assets/${ID}" >> "${FILE}" || true
    NEXT="$(stat -c %s "${FILE}")"
    if (( NEXT > EXPECTED )); then
      echo "Range response exceeded advertised size; inspect ${FILE}" >&2
      exit 2
    fi
    if (( NEXT == CURRENT )); then
      STALLED=$((STALLED + 1))
      if (( STALLED >= 5 )); then
        echo "Download stalled after five retries: ${NAME}" >&2
        exit 1
      fi
      sleep 3
    else
      STALLED=0
    fi
    CURRENT="${NEXT}"
  done
done

SUM_ID="$(gh api "repos/${REPO}/releases/tags/${TAG}" \
  --jq '.assets[] | select(.name == "SHA256SUMS") | .id')"
if [[ -z "${SUM_ID}" ]]; then
  echo "Release SHA256SUMS asset missing" >&2
  exit 2
fi
gh api -H 'Accept: application/octet-stream' \
  "repos/${REPO}/releases/assets/${SUM_ID}" > "${DEST}/SHA256SUMS.new"
mv "${DEST}/SHA256SUMS.new" "${DEST}/SHA256SUMS"
(
  cd "${DEST}"
  sha256sum -c SHA256SUMS
)
echo "All seven Stage-3 checkpoint parts verified in ${DEST}"
