#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="utt_test spk_test valid"

asr_config=conf/tuning/train_asr_hubert_bert_transformer_adam_specaug_finetune_gigaspeech.yaml

./asr_transcript.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --stage 1\
    --stop_stage 1 \
    --use_transcript true\
    --pretrained_model exp/asr_train_asr_raw_en_bpe5000/valid.acc.ave_10best.pth:encoder:encoder\
    --feats_normalize utterance_mvn\
    --nbpe 5000 \
    --token_type word\
    --audio_format wav\
    --feats_type raw\
    --max_wav_duration 30 \
    --inference_nj 8 \
    --nj 8 \
    --inference_asr_model valid.acc.ave_5best.pth\
    --asr_config "${asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"
