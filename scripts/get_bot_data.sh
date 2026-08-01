#!/usr/bin/env bash
# Fetch the datasets for the `botsage` replication of Deshmukh (2025),
# "Bot Detection in Social Media using GraphSage and BERT" (SJSU MS Project 1465).
#
# The paper evaluates on Cresci-15 and TwiBot-22. Neither can be downloaded
# without applying to its authors, so we use what is openly published:
#
#   cresci-2015  COMPLETE, from the TwiBot-22 authors' own conversion of the Bot
#                Repository datasets into the TwiBot-22 four-file schema. Has
#                node.json (users + tweets), edge.csv, label.csv, split.csv --
#                every ingredient the method needs.
#   twibot-22    PARTIAL: user.json + label.csv + split.csv are open on Zenodo
#                (record 7012904). edge.csv and the tweet files are not, so the
#                graph and text branches cannot run on it.
#   twibot-20    A preprocessed BotRGCN-format mirror: graph, 5 numeric user
#                properties, 768-d BERT tweet embeddings, labels. Stands in for
#                TwiBot-22 where the full pipeline is needed.
#
# Total ~2.5 GB. Respect each dataset's licence and the Twitter Developer
# Agreement; research use only.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw/bot"
mkdir -p "$RAW"

DRIVE_ID=1flNklDJG8wrv4oj5JHA9WAtV6JeGi1dW    # Other-Dataset-TwiBot22-Format.zip
ZENODO=https://zenodo.org/records/7012904/files
HF=https://huggingface.co/datasets/Sanjana7787/twibot-20-dataset/resolve/main/dataset

echo "==> Cresci-2015 (TwiBot-22 schema, ~770 MB archive)"
if [ ! -f "$RAW/cresci-2015/node.json" ]; then
    curl -L --progress-bar -o "$RAW/other.zip" \
        "https://drive.usercontent.google.com/download?id=$DRIVE_ID&export=download&confirm=t"
    unzip -oq "$RAW/other.zip" "Other-Dataset-TwiBot22-Format/cresci-2015/*" -d "$RAW/_tmp"
    mv "$RAW/_tmp/Other-Dataset-TwiBot22-Format/cresci-2015" "$RAW/"
    rm -rf "$RAW/_tmp" "$RAW/other.zip"
fi
printf '    %s labelled users\n' \
    "$(($(wc -l < "$RAW/cresci-2015/label.csv") - 1))"

echo "==> TwiBot-22 labels and split (Zenodo, open access)"
for f in label.csv split.csv; do
    [ -f "$RAW/twibot-22/$f" ] || {
        mkdir -p "$RAW/twibot-22"
        curl -L --progress-bar -o "$RAW/twibot-22/$f" "$ZENODO/$f?download=1"
    }
done
printf '    %s labelled users (%s bot)\n' \
    "$(($(wc -l < "$RAW/twibot-22/label.csv") - 1))" \
    "$(tail -n +2 "$RAW/twibot-22/label.csv" | grep -c bot)"

echo "==> TwiBot-22 user.json (782 MB) -- the five node features"
[ -f "$RAW/twibot-22/user.json" ] || \
    curl -L --progress-bar -o "$RAW/twibot-22/user.json" "$ZENODO/user.json?download=1"

echo "==> TwiBot-20 preprocessed (BotRGCN format, ~1.4 GB)"
mkdir -p "$RAW/twibot-20"
for f in label.pt num_properties_tensor.pt edge_index.pt split_new.json \
         tweets_tensor.pt; do
    [ -f "$RAW/twibot-20/$f" ] || \
        curl -L --progress-bar -o "$RAW/twibot-20/$f" "$HF/$f"
done

echo "done -> $RAW"
echo
echo "Next: precompute the text embeddings (Sect. 3.2), which is the slow stage:"
echo "  P=01-info-propagation/bot-detection-paper/.venv/bin/python"
echo "  \$P scripts/prepare_bot_embeddings.py --model distilbert-base-uncased"
echo "  \$P scripts/prepare_bot_embeddings.py --model bert-base-uncased"
