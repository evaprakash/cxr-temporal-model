#!/bin/bash
# Recompute every main-paper number from the saved score files. No GPU, no model
# loading: each step only re-reads npz / csv scores and reruns the metrics and the
# paired bootstrap. Run on ccw-gpu-1 (where /nvme lives) from the main cluster folder:
#
#   cd /nvme/people/eprakash/cxr-temporal-model && bash vljepa/check_paper_numbers.sh
#
# Everything goes to one log; send it back to compare against the manuscript.
set -uo pipefail

M="${MAIN_DIR:-/nvme/people/eprakash/cxr-temporal-model}"
D="${DYNAMIC_DIR:-/nvme/people/eprakash/cxr-temporal-model-dynamic}"
L="${LATERALITY_DIR:-/nvme/people/eprakash/cxr-temporal-model-laterality}"
cd "$M"

export PYTHONPATH="$M:$M/tempcxr/modules/hi-ml/hi-ml-multimodal/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PY="/nvme/people/eprakash/miniconda3/envs/roentgen/bin/python"
[ -x "$PY" ] || PY="$(command -v python || command -v python3)"
LOG="${CHECK_LOG:-/nvme/people/eprakash/logs/check_paper_numbers_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"

first() {  # first path that exists
    for p in "$@"; do [ -e "$p" ] && { echo "$p"; return; }; done
    echo "$1"
}

S=baap_eval/scores
SUP=$M/checkpoints_supervised_progression_unfrozen/class_signal_epoch_5.npz
VJ=$M/checkpoints_vljepa_lora_5way/class_signal_epoch_5.npz
VJ_SHIFT=$(first $M/vljepa/results/checkpoints_vljepa_lora_5way/epoch_5.shift.json \
                 $M/checkpoints_vljepa_lora_5way/epoch_5.shift.json)
FT=$M/checkpoints_vljepa_lora_5way_imgft/class_signal_epoch_4.npz
FT_SHIFT=$(first $M/vljepa/results/checkpoints_vljepa_lora_5way_imgft/epoch_4.shift.json \
                 $M/checkpoints_vljepa_lora_5way_imgft/epoch_4.shift.json)
DYN=$D/checkpoints_vljepa_dynamic/gold_vljepa_dyn_perfinding_epoch_9.npz
DYN_SHIFT=$D/vljepa/results/checkpoints_vljepa_dynamic/epoch_9.shift.json
DYNI=$D/checkpoints_vljepa_dynamic_img/gold_vljepa_dyn_perfinding_epoch_5.npz
DYNI_SHIFT=$D/vljepa/results/checkpoints_vljepa_dynamic_img/epoch_5.shift.json
LAT=$L/checkpoints_vljepa_lora_laterality/class_signal_epoch_5.npz
LAT_SHIFT=$L/vljepa/results/checkpoints_vljepa_lora_laterality/epoch_5.shift.json
LATI=$L/checkpoints_vljepa_lora_laterality_imgft/class_signal_epoch_5.npz
LATI_SHIFT=$L/vljepa/results/checkpoints_vljepa_lora_laterality_imgft/epoch_5.shift.json

G="$PY -m grounding_eval.eval_grounding compare"
ORTH="biovilt_all vljepa_enc_all vljepa_ft_enc_all"
CW4="biovilt_all_cw4 vljepa_enc_all_cw4 vljepa_ft_enc_all_cw4"
BASE="tila_all biovil_all biomedclip_all alta_all"
BASE_CW4="tila_all_cw4 biovil_all_cw4 biomedclip_all_cw4 alta_all_cw4"

section() { echo; echo "################ $* ################"; }

{
    echo "[check] main=$M dynamic=$D laterality=$L python=$PY"

    section "0. Input files"
    missing=0
    for f in $SUP $VJ $VJ_SHIFT $FT $FT_SHIFT $DYN $DYN_SHIFT $DYNI $DYNI_SHIFT \
             $LAT $LAT_SHIFT $LATI $LATI_SHIFT \
             $S/gold_biovilt.npz $S/gold_tila.npz $S/gold_biovil.npz $S/gold_biomedclip.npz $S/gold_alta.npz; do
        if [ -e "$f" ]; then echo "ok       $f"; else echo "MISSING  $f"; missing=1; fi
    done
    for n in $ORTH $BASE; do for m in text_orth text_presence; do
        f=grounding_eval/out/${n}_${m}.csv; [ -e "$f" ] || { echo "MISSING  $f"; missing=1; }
    done; done
    for n in $CW4 $BASE_CW4; do
        f=grounding_eval/out/${n}_text_orth.csv; [ -e "$f" ] || { echo "MISSING  $f"; missing=1; }
    done
    for n in biovilt vljepa_ft_btxt supervised_btxt; do
        f=grounding_eval/out/${n}_text.csv; [ -e "$f" ] || { echo "MISSING  $f"; missing=1; }
    done
    [ "$missing" = 0 ] && echo "all inputs present"

    section "1. Gold table (combined, macro F1, per-class F1; bootstrap vs fine-tuned 5-way)"
    $PY -m baap_eval.eval_baap compare-gold --baap none --ref vljepa_ft \
        --model biovil=$S/gold_biovil.npz --model biomedclip=$S/gold_biomedclip.npz \
        --model biovilt=$S/gold_biovilt.npz --model tila=$S/gold_tila.npz --model alta=$S/gold_alta.npz \
        --model supervised=$SUP \
        --model vljepa=$VJ:$VJ_SHIFT --model vljepa_ft=$FT:$FT_SHIFT \
        --model vljepa_dyn=$DYN:$DYN_SHIFT --model vljepa_dyn_img=$DYNI:$DYNI_SHIFT \
        --model vljepa_lat=$LAT:$LAT_SHIFT --model vljepa_lat_ft=$LATI:$LATI_SHIFT \
        --model vljepa_lat_raw=$LAT --model vljepa_lat_ft_raw=$LATI

    section "1b. Gold: fine-tuned 5-way vs supervised (text: +.023 combined, +.032 accuracy, CheXpert +.031)"
    $PY -m baap_eval.eval_baap compare-gold --baap none --ref supervised \
        --model supervised=$SUP --model vljepa_ft=$FT:$FT_SHIFT

    section "2. MS-CXR-T (339 pairs)"
    $PY -m vljepa.eval_mscxrt compare --out-dir $M/mscxrt_scores_patient --ref vljepa_ft
    $PY -m vljepa.eval_mscxrt compare --out-dir $M/mscxrt_scores_patient --ref supervised

    section "3. Chest ImaGenome (348 pairs)"
    $PY -m vljepa.eval_mscxrt compare --out-dir $M/cig_scores_patient_ours --ref vljepa_ft
    $PY -m vljepa.eval_mscxrt compare --out-dir $M/cig_scores_patient_ours --ref supervised

    section "4. Grounding: progression (text orth), one wording -- Fig. 6a"
    $G --method text_orth --names $ORTH --ref biovilt_all
    section "5. Grounding: progression, four paraphrases -- Fig. 6b"
    $G --method text_orth --names $CW4 --ref biovilt_all_cw4
    section "6. Grounding: Worse annotations only -- Fig. 6a,b last group"
    $G --method text_orth --names $ORTH --ref biovilt_all --label worsening
    $G --method text_orth --names $CW4 --ref biovilt_all_cw4 --label worsening
    section "7. Grounding: finding presence -- Fig. 6c"
    $G --method text_presence --names $ORTH --ref biovilt_all
    section "8. Grounding: image encoders with the BioViL-T text encoder -- Fig. 6d"
    $G --method text --names biovilt vljepa_ft_btxt supervised_btxt --ref biovilt
    section "9. Grounding: Resolved (text: BioViL-T .459, Prog-JEPA .325)"
    $G --method text_orth --names $ORTH --ref vljepa_ft_enc_all --label resolved
    section "10. Grounding: external baselines (text: TILA .565 orth, .634 presence)"
    $G --method text_orth --names $ORTH $BASE --ref vljepa_ft_enc_all
    $G --method text_presence --names $ORTH $BASE --ref vljepa_ft_enc_all
    $G --method text_orth --names $CW4 $BASE_CW4 --ref vljepa_ft_enc_all_cw4
} 2>&1 | tee "$LOG"

echo
echo "[check] log: $LOG"
