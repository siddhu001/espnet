#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

src_type=en
tgt_type=en

train_set="train"
valid_set="valid"
test_sets="test"

nlu_config=conf/train_nlu2.yaml
inference_config=conf/decode_nlu2.yaml

src_case=asr
tgt_case=ner

src_nbpe=500
tgt_nbpe=500   # if token_joint is true, then only tgt_nbpe is used

./nlu.sh \
    --ngpu 1 \
    --nj 16 \
    --inference_nj 32 \
    --stage 10\
    --stop_stage 10\
    --src_type ${src_type} \
    --tgt_type ${tgt_type} \
    --src_token_type "bpe" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "bpe" \
    --tgt_nbpe $tgt_nbpe \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --feats_type raw \
    --nlu_config "${nlu_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --src_bpe_train_text "data/${train_set}/bpe_text" \
    --tgt_bpe_train_text "data/${train_set}/bpe_text" \
    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_type}" "$@"
