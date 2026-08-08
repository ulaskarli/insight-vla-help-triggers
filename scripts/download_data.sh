#!/usr/bin/env bash
set -euo pipefail
cat <<'EOF'
INSIGHT datasets are not bundled with the Git repository.
See docs/DATA_FORMAT.md for the expected processed format.
If public processed-data assets are released, this script will be updated with
their versioned download locations and checksums.
EOF
