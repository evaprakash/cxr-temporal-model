#!/bin/bash
# Re-score a few main-paper configurations from their checkpoints (GPU) and compare
# them against the saved score files the paper tables were built from. Identical
# numbers (up to GPU nondeterminism, ~1e-4) mean the saved scores are reproducible.
#
#   Fine-tuned Prog-JEPA, supervised, BioViL-T  on gold, MS-CXR-T, Chest ImaGenome
#   BioViL-T, fine-tuned Prog-JEPA              grounding, text_orth and text_presence
#
# Run on ccw-gpu-1 inside an srun shell with one GPU:
#
#   cd /nvme/people/eprakash/cxr-temporal-model && bash vljepa/recheck_paper_numbers.sh
#
# New scores go to recheck_<timestamp>/ and grounding_eval/out/recheck_*; nothing
# the paper used is overwritten.
set -uo pipefail

M="${MAIN_DIR:-/nvme/people/eprakash/cxr-temporal-model}"
cd "$M"
export SCRATCH_BASE="${SCRATCH_BASE:-/nvme/people/eprakash}"
export PROJECT_DIR="$M"
export JEPA_IMAGE_ROOTS_DIR="${JEPA_IMAGE_ROOTS_DIR:-/nvme/data/temporal/all_data}"
export VLJEPA_HF_HOME="${VLJEPA_HF_HOME:-$SCRATCH_BASE/hf}"
export HF_HOME="$VLJEPA_HF_HOME"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$VLJEPA_HF_HOME/hub}"
export VLJEPA_LLAMA_LOCAL="${VLJEPA_LLAMA_LOCAL:-$VLJEPA_HF_HOME/Llama-3.2-1B}"
export PYTHONPATH="$M:$M/tempcxr/modules/hi-ml/hi-ml-multimodal/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PY="/nvme/people/eprakash/miniconda3/envs/roentgen/bin/python"
[ -x "$PY" ] || PY="$(command -v python || command -v python3)"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$M/recheck_$STAMP"
mkdir -p "$OUT"/{gold,mscxrt,cig}
LOG="$SCRATCH_BASE/logs/recheck_paper_numbers_$STAMP.log"
mkdir -p "$(dirname "$LOG")"

SUP_CKPT=checkpoints_supervised_progression_unfrozen/epoch_5.pt
FT_CKPT=checkpoints_vljepa_lora_5way_imgft/epoch_4.pt
FT_SHIFTED=checkpoints_vljepa_lora_5way_imgft/epoch_4_shifted.pt
FT_SHIFT=vljepa/results/checkpoints_vljepa_lora_5way_imgft/epoch_4.shift.json
[ -e "$FT_SHIFT" ] || FT_SHIFT=checkpoints_vljepa_lora_5way_imgft/epoch_4.shift.json
G="$PY -m grounding_eval.eval_grounding"

section() { echo; echo "################ $* ################"; }

{
    echo "[recheck] out=$OUT python=$PY"
    for f in $SUP_CKPT $FT_CKPT $FT_SHIFTED $FT_SHIFT mscxrt_clean_patient.csv cig_clean_patient_ours.csv; do
        [ -e "$f" ] && echo "ok       $f" || echo "MISSING  $f"
    done

    section "1. Gold: re-score"
    # Gold is what we check; a small silver-val slice keeps the file format without the full val pass.
    $PY -m vljepa.check_class_signal --ckpt $FT_CKPT --out $OUT/gold/vljepa_ft.npz \
        --val-limit 256 --no-full-report
    $PY -m vljepa.check_class_signal --supervised --ckpt $SUP_CKPT --out $OUT/gold/supervised.npz \
        --val-limit 256 --no-full-report
    $PY -m vljepa.score_gold_baselines --backend biovilt --out-dir $OUT/gold --name biovilt

    section "1b. Gold: saved vs re-scored (differences should be ~0)"
    $PY -m baap_eval.eval_baap compare-gold --baap none --ref vljepa_ft \
        --model vljepa_ft=checkpoints_vljepa_lora_5way_imgft/class_signal_epoch_4.npz:$FT_SHIFT \
        --model vljepa_ft_NEW=$OUT/gold/vljepa_ft.npz:$FT_SHIFT \
        --model supervised=checkpoints_supervised_progression_unfrozen/class_signal_epoch_5.npz \
        --model supervised_NEW=$OUT/gold/supervised.npz \
        --model biovilt=baap_eval/scores/gold_biovilt.npz \
        --model biovilt_NEW=$OUT/gold/gold_biovilt.npz

    for SET in mscxrt cig; do
        if [ $SET = mscxrt ]; then CLEAN=mscxrt_clean_patient.csv; OLD=mscxrt_scores_patient
        else CLEAN=cig_clean_patient_ours.csv; OLD=cig_scores_patient_ours; fi
        section "2. $SET: re-score"
        $PY -m vljepa.eval_mscxrt score --model vljepa --ckpt $FT_SHIFTED --name vljepa_ft_NEW \
            --clean $CLEAN --out-dir $OUT/$SET
        $PY -m vljepa.eval_mscxrt score --model supervised --ckpt $SUP_CKPT --name supervised_NEW \
            --clean $CLEAN --out-dir $OUT/$SET
        $PY -m vljepa.eval_mscxrt score --model biovilt --name biovilt_NEW \
            --clean $CLEAN --out-dir $OUT/$SET
        section "2b. $SET: saved vs re-scored"
        $PY -m vljepa.eval_mscxrt compare --ref vljepa_ft --npz \
            $OLD/vljepa_ft.npz $OLD/supervised.npz $OLD/biovilt.npz \
            $OUT/$SET/vljepa_ft_NEW.npz $OUT/$SET/supervised_NEW.npz $OUT/$SET/biovilt_NEW.npz
    done

    section "3. Grounding: re-score maps"
    for MTH in text_orth text_presence; do
        $G maps --model biovilt --name recheck_biovilt_all --presence all --method $MTH
        $G maps --model vljepa --name recheck_vljepa_ft_enc_all --vljepa-text encoder \
            --ckpt $FT_SHIFTED --presence all --method $MTH
    done
    section "3b. Grounding: saved vs re-scored"
    for MTH in text_orth text_presence; do
        $G compare --method $MTH --ref biovilt_all \
            --names biovilt_all recheck_biovilt_all vljepa_ft_enc_all recheck_vljepa_ft_enc_all
    done
    $G compare --method text_orth --ref biovilt_all --label worsening \
        --names biovilt_all recheck_biovilt_all vljepa_ft_enc_all recheck_vljepa_ft_enc_all
} 2>&1 | tee "$LOG"

echo
echo "[recheck] log: $LOG"
