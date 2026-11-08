#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# wise_mosaics.sh
#
# Wrapper for casda/wise_mosaics.py that:
#   - Must be run from the RGCAT repo root
#   - Accepts an optional --config path (default: config/pipeline.yaml)
#   - Cleans up any stale per-tile scratch directories left by a crash
#   - Retries automatically on failure (wise_mosaics.py exits 1 on partial
#     failure; already-finished mosaics are skipped on the next attempt)
#   - Stops retrying after MAX_RETRIES attempts
#
# Usage:
#   bash casda/wise_mosaics.sh [--config config/pipeline.yaml]
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="config/pipeline.yaml"
MAX_RETRIES=5

# Parse optional --config argument
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"; shift 2 ;;
        --config=*)
            CONFIG="${1#*=}"; shift ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Read wise_images_dir from pipeline.yaml (fallback to default)
WISE_DIR=$(python - <<'PYEOF'
import sys, yaml
try:
    cfg = yaml.safe_load(open("${CONFIG}"))
    print(cfg.get("paths", {}).get("wise_images_dir", "data/images_wise"))
except Exception:
    print("data/images_wise")
PYEOF
)
# Re-expand CONFIG variable in the heredoc result
WISE_DIR=$(python -c "
import yaml
cfg = yaml.safe_load(open('${CONFIG}'))
print(cfg.get('paths', {}).get('wise_images_dir', 'data/images_wise'))
")

echo "================================================================"
echo " WISE mosaic generation"
echo "  Repo root : ${REPO_ROOT}"
echo "  Config    : ${CONFIG}"
echo "  Output dir: ${WISE_DIR}"
echo "================================================================"

# ---------------------------------------------------------------------------
# Cleanup: remove any leftover per-tile scratch directories (but not mosaics)
# These are directories named *_wise/ (not ending in _wise_mosaic.fits)
# ---------------------------------------------------------------------------
cleanup_scratch_dirs() {
    local wise_dir="${REPO_ROOT}/${WISE_DIR}"
    if [[ ! -d "${wise_dir}" ]]; then return; fi

    local removed=0
    while IFS= read -r -d '' d; do
        echo "  Removing stale scratch dir: ${d}"
        rm -rf "${d}"
        (( removed++ )) || true
    done < <(find "${wise_dir}" -maxdepth 1 -type d -name '*_wise' -print0)

    if [[ "${removed}" -gt 0 ]]; then
        echo "  Removed ${removed} stale scratch directory/directories."
    fi
}

# ---------------------------------------------------------------------------
# Main retry loop
# ---------------------------------------------------------------------------
attempt=0
while true; do
    (( attempt++ )) || true

    echo ""
    echo "--- Attempt ${attempt} / ${MAX_RETRIES} ---"

    if python casda/wise_mosaics.py --config "${CONFIG}"; then
        echo ""
        echo "wise_mosaics.py completed successfully."
        cleanup_scratch_dirs
        exit 0
    fi

    exit_code=$?
    echo ""
    echo "wise_mosaics.py exited with code ${exit_code}."

    # Clean up any scratch dirs left by the partial run before retrying
    cleanup_scratch_dirs

    if [[ "${attempt}" -ge "${MAX_RETRIES}" ]]; then
        echo "Reached maximum retries (${MAX_RETRIES}). Giving up."
        exit 1
    fi

    echo "Retrying in 5 seconds..."
    sleep 5
done
