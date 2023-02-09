#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="test_asr_whisper"

asr_config=conf/train_asr2_wavlm_bart3_loss.yaml

./slu.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --nbpe 500 \
    --use_transcript true \
    --token_type bpe\
    --feats_type raw\
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn\
    --inference_nj 10 \
    --inference_slu_model valid.acc.ave_10best.pth\
    --slu_config "${asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --use_lm true\
    --lm_config conf/train_rnn_lm.yaml\
    --speed_perturb_factors "0.9 1.0 1.1" \
    --nj 6\
    --gpu_inference true \
    --bpe_train_text data/train/bpe_text \
    --bpe_train_transcript data/train/bpe_text \
    --inference_config conf/decode_asr2.yaml\
    --lm_train_text dump/raw/train_sp/text \
    --test_sets "${test_sets}" "$@"
