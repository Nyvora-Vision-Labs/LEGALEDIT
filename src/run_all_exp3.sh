#!/bin/bash
# Full exp3 programme: the document-grouped split (Table S9 of the
# supplement) first, then the random split (Table S7). Both stages are resumable -- run_exp3_grid.sh skips
# any (config, seed) already banked -- so this can be re-launched at any point.
#
# Grouped runs first because Table S9 is the table with no fallback: Table S7
# has banked seeds to show, Table S9 has nothing until its three configs finish.
set -u
cd "$(dirname "$0")"

# Grouped split: the three configurations the generalization comparison needs
# (the reproduction, the original augmentation, and the LegalEdit-augmented
# variant). The remaining heads differ in training target, not in how they
# would respond to a change of source document.
CONFIGS='JudgeBERT (repro)|JudgeBERT-DA (repro)|JudgeBERT-DA+LegalEdit' \
GROUPED=1 TAG=_grouped ./run_exp3_grid.sh

./run_exp3_grid.sh
