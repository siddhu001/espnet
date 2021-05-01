#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
stage=2        # start from -1 if you need to start from data download
stop_stage=100
nj=8            # number of parallel jobs for decoding
dumpdir=dump    # directory to dump full features
verbose=0       # verbose option

# bpemode (unigram or bpe)
nbpe=2000

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

dataset="sw-fr"
src_lang=sw
tgt_lang=fr
src_case=lc.rm
tgt_case=lc.rm

feat_tr_dir=dump/mt/
mkdir -p ${feat_tr_dir}

dict=data/${dataset}-all/bpe${nbpe}_units.txt
nlsyms=data/${dataset}-all/non_lang_syms.txt
bpemodel=data/${dataset}-all/bpe${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    data2json.sh --nj 16 --text data/${dataset}/text.${tgt_case}.${tgt_lang} --bpecode ${bpemodel}.model \
        data/${dataset} ${dict} > ${feat_tr_dir}/data_bpe${nbpe}.${src_case}_${tgt_lang}.json

    update_json.sh --text data/${dataset}/text.${src_case}.${src_lang} --bpecode ${bpemodel}.model \
        ${feat_tr_dir}/data_bpe${nbpe}.${src_case}_${tgt_lang}.json data/${dataset} ${dict}
    head -n 1000 < data/${dataset}/text.${tgt_case}.${tgt_lang} > data/${dataset}/text.${tgt_case}.${tgt_lang}.1000
    head -n 1000 < data/${dataset}/text.${src_case}.${src_lang} > data/${dataset}/text.${src_case}.${src_lang}.1000
    data2json.sh --nj 16 --text data/${dataset}/text.${tgt_case}.${tgt_lang}.1000 --bpecode ${bpemodel}.model \
        data/${dataset} ${dict} > ${feat_tr_dir}/data_dt_bpe${nbpe}.${src_case}_${tgt_lang}.json

    update_json.sh --text data/${dataset}/text.${src_case}.${src_lang}.1000 --bpecode ${bpemodel}.model \
        ${feat_tr_dir}/data_dt_bpe${nbpe}.${src_case}_${tgt_lang}.json data/${dataset} ${dict}
fi
