#!/usr/bin/env bash
# Download POV-Surgery from the public project page.
#
# POV-Surgery is publicly distributed at https://batfacewayne.github.io/POV_Surgery_io/.
# Follow the page's instructions to obtain the dataset; this script is a
# placeholder that documents the expected on-disk layout.
#
# Usage: scripts/download_pov.sh [DEST_DIR]
#         (defaults to data/pov_surgery)
set -euo pipefail

DEST="${1:-data/pov_surgery}"
mkdir -p "$DEST"

cat <<EOF
POV-Surgery is publicly available at:
  https://batfacewayne.github.io/POV_Surgery_io/

After acquiring the dataset, the expected layout under "$DEST" is:

  $DEST/
    POV_Surgery_info.csv
    annotation/
    handoccnet_train/
    tool_mesh/
    mask/
    mask_blender/
    color/

Once the data is in place, run:
  make eval-wilor-pov DATA="\$(dirname $DEST)"
EOF
