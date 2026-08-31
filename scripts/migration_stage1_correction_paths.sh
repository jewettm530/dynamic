#!/usr/bin/env bash
set -euo pipefail

# Idempotent helper for an existing repository that still has the old Stage 1
# result/config/output paths. Run from repository root before committing the
# correction. The local output/ moves are especially useful because output/
# was intentionally omitted from the project zip supplied for this correction.

mkdir -p \
  results/stage1/T0_two_frame_oracle \
  results/stage1/archive_pre_correction \
  configs/t0_two_frame_oracle \
  configs/archive_pre_correction \
  output/stage1/T0_two_frame_oracle \
  output/stage1/archive_pre_correction

move_if_present() {
  local src="$1" dst="$2"
  if [ -e "$src" ] && [ ! -e "$dst" ]; then
    mkdir -p "$(dirname "$dst")"
    mv "$src" "$dst"
    echo "moved: $src -> $dst"
  elif [ -e "$src" ] && [ -e "$dst" ]; then
    echo "kept both (destination already exists): $src ; $dst"
  fi
}

# Tracked summaries/configurations.
move_if_present results/stage1/weighting results/stage1/T0_two_frame_oracle/weighting
move_if_present results/stage1/final_controlled results/stage1/T0_two_frame_oracle/final_controlled
move_if_present results/stage1/comparison_artifact results/stage1/archive_pre_correction/unmatched_comparison

for f in multitask_w1.yaml multitask_w2.yaml multitask_w3.yaml multitask_selected.yaml ef_only_controlled.yaml segmentation_only_controlled.yaml; do
  move_if_present "configs/$f" "configs/t0_two_frame_oracle/$f"
done
for f in ef_only.yaml segmentation_only.yaml; do
  move_if_present "configs/$f" "configs/archive_pre_correction/$f"
done

# Local run artifacts/checkpoints omitted from the uploaded zip. These are not
# Git-tracked, but moving them prevents T0 and corrected B1/B2/B3 from sharing
# ambiguous Stage 1 folder names.
move_if_present output/stage1/weighting output/stage1/T0_two_frame_oracle/weighting
move_if_present output/stage1/final_baselines_controlled output/stage1/T0_two_frame_oracle/final_baselines_controlled
move_if_present output/stage1/final_evaluation_controlled output/stage1/T0_two_frame_oracle/final_evaluation_controlled
move_if_present output/stage1/final_baselines output/stage1/archive_pre_correction/final_baselines
move_if_present output/stage1/final_evaluation output/stage1/archive_pre_correction/final_evaluation

# Historical audit artifacts. The two-frame alignment examples are patient-
# derived visualizations, so they should not be committed under the professor's
# reporting rule.
move_if_present outputs/stage1_audit_final outputs/stage1/T0_two_frame_oracle/audit_final
rm -rf outputs/stage1_audit_final/alignment_examples \
       outputs/stage1/T0_two_frame_oracle/audit_final/alignment_examples 2>/dev/null || true

cat <<'TXT'

Stage 1 historical paths are organized.
Corrected runs now belong under:
  output/stage1/corrected/
  results/stage1/corrected/

Review `git status` before committing. Do not add checkpoints, videos, masks,
tracings, or patient-derived visualizations to Git.
TXT
