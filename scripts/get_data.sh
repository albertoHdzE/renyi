#!/usr/bin/env bash
# Fetch the two datasets named in the paper's Data Availability Statement.
#
#   CollegeMsg          https://snap.stanford.edu/data/index.html
#   Chinese rumour data https://github.com/yeren66/ChineseRumorDataset
#
# The second link is an *index* of eight corpora rather than a dataset. Only the
# CED subset carries the repost/comment structure the method needs, so we clone
# the upstream thunlp repository it points to. See docs/DISCREPANCIES.md §9.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"

echo "==> CollegeMsg (SNAP)"
if [ ! -f "$RAW/CollegeMsg.txt" ]; then
    curl -sL -o "$RAW/CollegeMsg.txt.gz" https://snap.stanford.edu/data/CollegeMsg.txt.gz
    gunzip -kf "$RAW/CollegeMsg.txt.gz"
fi
printf '    %s edges\n' "$(wc -l < "$RAW/CollegeMsg.txt" | tr -d ' ')"

echo "==> Chinese rumour dataset index"
[ -d "$RAW/ChineseRumorDataset" ] || \
    git clone --depth 1 -q https://github.com/yeren66/ChineseRumorDataset.git \
        "$RAW/ChineseRumorDataset"

echo "==> CED cascades (thunlp, ~283 MB)"
[ -d "$RAW/thunlp_rumor" ] || \
    git clone --depth 1 -q https://github.com/thunlp/Chinese_Rumor_Dataset.git \
        "$RAW/thunlp_rumor"
printf '    %s rumour cascades\n' "$(ls "$RAW/thunlp_rumor/CED_Dataset/rumor-repost" | wc -l | tr -d ' ')"

echo "done -> $RAW"
