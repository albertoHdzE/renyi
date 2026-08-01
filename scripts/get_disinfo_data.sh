#!/usr/bin/env bash
# Fetch the four datasets used by the `disinfo` replication of Lakzaei,
# Haghir Chehreghani & Bagheri (2024), "Disinformation detection using graph
# neural networks: a survey", Artificial Intelligence Review 57:52.
#
# All four appear in the survey's Table 3. Sizes printed here are checked
# against that table by ``disinfo.survey_data.verify_table3``.
#
#   LIAR              Table 3 says 12,836 statements, 6 labels   (Wang 2017)
#   Twitter15/16      Table 3 says 1490 / 818 trees              (Ma et al. 2017)
#   PHEME             Table 3 says 6425 tweets, 4 labels         (Zubiaga 2016)
#   CED (Sina Weibo)  Table 3 says 4664 claims, 2 labels         (Ma et al. 2016)
#
# CED is already fetched by ``scripts/get_data.sh`` for the entropia paper and
# is reused from ``data/raw/thunlp_rumor`` rather than downloaded twice.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"

echo "==> LIAR (politifact.com, 6-class)"
if [ ! -f "$RAW/liar/train.tsv" ]; then
    curl -sL -o "$RAW/liar.zip" https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
    unzip -oq "$RAW/liar.zip" -d "$RAW/liar"
    rm -f "$RAW/liar.zip"
fi
printf '    %s statements\n' \
    "$(cat "$RAW"/liar/{train,valid,test}.tsv | wc -l | tr -d ' ')"

echo "==> Twitter15 + Twitter16 (Ma et al. 2017 propagation trees)"
if [ ! -d "$RAW/rumor_detection_acl2017" ]; then
    curl -sL -o "$RAW/rumdetect2017.zip" \
        "https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=1"
    unzip -oq "$RAW/rumdetect2017.zip" -d "$RAW"
    rm -f "$RAW/rumdetect2017.zip"
fi
printf '    twitter15: %s trees\n' \
    "$(wc -l < "$RAW/rumor_detection_acl2017/twitter15/label.txt" | tr -d ' ')"
printf '    twitter16: %s trees\n' \
    "$(wc -l < "$RAW/rumor_detection_acl2017/twitter16/label.txt" | tr -d ' ')"

echo "==> PHEME veracity (figshare 6392078, ~46 MB)"
if [ ! -d "$RAW/pheme" ]; then
    curl -sL -o "$RAW/pheme.tar.bz2" \
        https://ndownloader.figshare.com/files/11767817
    mkdir -p "$RAW/pheme"
    tar -xjf "$RAW/pheme.tar.bz2" -C "$RAW/pheme"
    rm -f "$RAW/pheme.tar.bz2"
fi
printf '    %s threads\n' \
    "$(find "$RAW/pheme" -name 'source-tweet*' -type d | wc -l | tr -d ' ')"

echo "==> CED / Sina Weibo cascades"
if [ ! -d "$RAW/thunlp_rumor" ]; then
    echo "    not present - running scripts/get_data.sh for it"
    bash "$ROOT/scripts/get_data.sh"
else
    printf '    %s rumour cascades (reused from get_data.sh)\n' \
        "$(ls "$RAW/thunlp_rumor/CED_Dataset/rumor-repost" | wc -l | tr -d ' ')"
fi

echo "done -> $RAW"
