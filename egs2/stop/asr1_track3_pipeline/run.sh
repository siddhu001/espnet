#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets_r="test-reminder"
test_sets_w="test-weather"

asr_config_r=conf/train_asr_whisper_full_correct_dup20r_lower_iter1k.yaml
asr_config_w=conf/train_asr_whisper_full_correct_dup20w_lower_iter1k.yaml

asr_stats_dir_r=exp/asr_stats_dup20r_lower_raw_en_whisper_multilingual_sp/
asr_stats_dir_w=exp/asr_stats_dup20w_lower_raw_en_whisper_multilingual_sp/

inference_asr_model=valid.acc.ave.pth
inference_asr_config=conf/decode_asr_whisper_noctc_greedy_v2.yaml

ngpu=1

./asr.sh \
    --stage 2 \
    --stop_stage 9 \
    --lang en \
    --ngpu ${ngpu} \
    --use_lm false \
    --lm_train_text data/train/text \
    --token_type whisper_multilingual \
    --feats_normalize '' \
    --feats_type raw\
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn\
    --inference_nj 16 \
    --nj 16 \
    --speed_perturb_factors "0.9 1.0 1.1"\
    --inference_asr_model "${inference_asr_model}" \
    --inference_config "${inference_asr_config}"\
    --asr_config "${asr_config_r}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"

# Prepare `dump/raw/test-{reminder/weather}` and `dump/raw/valid-{reminder/weather}-25spis`
bash local/split_rw.sh

# Train reminder ASR

./asr.sh \
    --stage 10 \
    --stop_stage 11 \
    --lang en \
    --ngpu ${ngpu} \
    --use_lm false \
    --lm_train_text data/train/text \
    --token_type whisper_multilingual \
    --feats_normalize '' \
    --feats_type raw\
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn\
    --inference_nj 1 \
    --nj 1 \
    --speed_perturb_factors "0.9 1.0 1.1"\
    --asr_config "${asr_config_r}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@" \
    --asr_stats_dir "${asr_stats_dir_r}"

# Train weather ASR

./asr.sh \
    --stage 10 \
    --stop_stage 11 \
    --lang en \
    --ngpu ${ngpu} \
    --use_lm false \
    --lm_train_text data/train/text \
    --token_type whisper_multilingual \
    --feats_normalize '' \
    --feats_type raw\
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn\
    --inference_nj 1 \
    --nj 1 \
    --speed_perturb_factors "0.9 1.0 1.1"\
    --asr_config "${asr_config_w}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@" \
    --asr_stats_dir "${asr_stats_dir_w}"

# Decode reminder ASR

./asr.sh \
    --stage 12 \
    --stop_stage 13 \
    --lang en \
    --ngpu ${ngpu} \
    --use_lm false \
    --lm_train_text data/train/text \
    --token_type whisper_multilingual \
    --feats_normalize '' \
    --feats_type raw\
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn\
    --inference_nj 24 \
    --nj 24 \
    --speed_perturb_factors "0.9 1.0 1.1"\
    --inference_asr_model "${inference_asr_model}" \
    --inference_config "${inference_asr_config}"\
    --asr_config "${asr_config_r}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets_r}" "$@"

# Decode weather ASR

./asr.sh \
    --stage 12 \
    --stop_stage 13 \
    --lang en \
    --ngpu ${ngpu} \
    --use_lm false \
    --lm_train_text data/train/text \
    --token_type whisper_multilingual \
    --feats_normalize '' \
    --feats_type raw\
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn\
    --inference_nj 24 \
    --nj 24 \
    --speed_perturb_factors "0.9 1.0 1.1"\
    --inference_asr_model "${inference_asr_model}" \
    --inference_config "${inference_asr_config}"\
    --asr_config "${asr_config_w}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets_w}" "$@"
