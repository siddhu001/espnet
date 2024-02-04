#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=MuST-C_v2_en-de/train
valid_set=MuST-C_v2_en-de/dev
test_sets="MuST-C_v2_en-de/dev"

nbpe=5000
bpe_nlsyms=data/nlsyms.txt
s2t_config=conf/train_s2t_ebf_e18_lr2e-4.yaml
inference_config=conf/decode_s2t.yaml

# inference only args
# --cleaner whisper_en --hyp_cleaner whisper_en
./s2t.sh \
    --stage 11 \
    --stop_stage 11 \
    --use_lm false \
    --num_nodes 1 \
    --ngpu 2 \
    --nj 32 \
    --gpu_inference true \
    --inference_nj 4 \
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
