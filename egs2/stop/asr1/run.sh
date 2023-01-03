#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="test valid"

asr_config=conf/train_asr2_wavlm_lr0.002.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --nbpe 500 \
    --token_type bpe\
    --feats_type raw\
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn\
    --bpe_train_text data/train/bpe_text\
    --inference_nj 8 \
    --inference_asr_model valid.acc.ave_10best.pth\
    --speed_perturb_factors "0.9 1.0 1.1"
    --pretrained_model "/ocean/projects/cis210027p/siddhana/new_download/espnet/egs2/stop/asr1_pipeline/exp/asr_train_asr2_wavlm_lr0.002_raw_en_bpe500_sp/valid.acc.ave_10best.pth"
    --asr_config "${asr_config}" \
    --inference_config conf/decode_asr.yaml \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"
