#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="utt_test spk_test valid"

asr_config=conf/tuning/train_asr_conformer_adam_specaug_fixed_gigaspeech.yaml

./asr_transcript.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --pretrained_model ../../fsc_challenge/asr1/exp/asr_train_asr_raw_en_bpe5000/valid.acc.ave_10best.pth:encoder:encoder\
    --stage 11 \
    --stop_stage 16 \
    --feats_normalize utterance_mvn\
    --nbpe 5000 \
    --token_type word\
    --audio_format wav\
    --feats_type raw\
    --max_wav_duration 30 \
    --inference_nj 8 \
    --nj 7 \
    --inference_asr_model valid.acc.ave_5best.pth\
    --asr_config "${asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"
