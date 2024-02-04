#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=owsm_train.en-de_sp
valid_set=owsm_dev.en-de
test_sets="tst-COMMON.en-de tst-HE.en-de"

nbpe=10000
bpe_nlsyms=data/nlsyms.txt

s2t_config=conf/train_s2t_finetune_stonly_ebf_e24_lr2e-3_warmup25k.yaml
inference_config=conf/decode_s2t.yaml

# inference only args
# --cleaner whisper_en --hyp_cleaner whisper_en
./s2t.sh \
    --stage 11 \
    --stop_stage 11 \
    --use_lm false \
    --num_nodes 2 \
    --ngpu 4 \
    --nj 32 \
    --gpu_inference true \
    --inference_nj 4 \
    --inference_s2t_model valid.loss_ctc.ave_10best.pth \
    --num_splits_s2t 1 \
    --feats_type raw \
    --audio_format flac.ark \
    --token_type bpe \
    --nbpe ${nbpe} \
    --bpe_input_sentence_size 15000000 \
    --s2t_config "${s2t_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "dump/raw/${train_set}/text" \
    --bpe_nlsyms ${bpe_nlsyms} \
    --lm_train_text "dump/raw/${train_set}/text" "$@"
