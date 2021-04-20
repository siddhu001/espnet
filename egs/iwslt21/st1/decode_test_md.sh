#!/usr/bin/env bash

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch # chainer or pytorch
stage=0         # start from -1 if you need to start from data download
stop_stage=5
ngpu=4          # number of gpus during training ("0" uses cpu, otherwise use gpu)
dec_ngpu=0      # number of gpus during decoding ("0" uses cpu, otherwise use gpu)
nj=12            # number of parallel jobs for decoding
debugmode=1
dumpdir=/scratch/iwslt_dump    # directory to dump full features
N=0             # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0       # verbose option
resume=         # Resume the training from snapshot
seed=1          # seed to generate random number
# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/train.yaml
decode_config=conf/decode.yaml

# decoding parameter
trans_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=5                  # the number of ST models to be averaged
use_valbest_average=false     # if true, the validation `n_average`-best ST models will be averaged.
                             # if false, the last `n_average` ST models will be averaged.
metric=bleu                  # loss/acc/bleu

# pre-training related
asr_model=
mt_model=

# preprocessing related
src_case=lc.rm
tgt_case=tc
# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal

# postprocessing related
remove_nonverbal=true  # remove non-verbal labels such as "( Applaus )"
# NOTE: IWSLT community accepts this setting and therefore we use this by default

# bpemode (unigram or bpe)
nbpe=16000
bpemode=bpe

# exp tag
tag="" # tag for managing experiments.

expdir=
part=
. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

#mkdir -p ${dumpdir}
#rsync -av --progress iwslt_dump/* ${dumpdir}/
#trap 'rm -rf ${dumpdir}' EXIT

# data directories
mustc_dir=../../must_c
mustc_v2_dir=../../must_c_v2
stted_dir=../../iwslt18
iwslt_test_data=/project/ocean/byan/corpora/iwslt18

train_set=train.de
train_dev=dev.de
#trans_subset="et_mustc_dev_org.de et_mustc_tst-COMMON.de et_mustc_tst-HE.de"
trans_subset="et_mustc_dev_org.de et_mustc_tst-COMMON.de"
#trans_set="et_mustc_dev_org.de et_mustc_tst-COMMON.de et_mustc_tst-HE.de \
#           et_mustcv2_dev_org.de et_mustcv2_tst-COMMON.de et_mustcv2_tst-HE.de \
#           et_stted_dev2010.de et_stted_tst2010.de et_stted_tst2013.de et_stted_tst2014.de et_stted_tst2015.de \
#           et_stted_tst2018.de et_stted_tst2019.de"
trans_set="et_stted_dev2010.de et_stted_tst2010.de et_stted_tst2013.de et_stted_tst2014.de et_stted_tst2015.de \
           et_stted_tst2018.de et_stted_tst2019.de"
trans_set1="et_stted_tst2018.de et_stted_tst2019.de"
trans_set2="et_stted_tst2010.de et_stted_tst2015.de"
trans_set3="et_stted_tst2013.de et_stted_tst2014.de"
feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
dict=data/lang_1spm/${train_set}_${bpemode}${nbpe}_units_${tgt_case}.txt
nlsyms=data/lang_1spm/${train_set}_non_lang_syms_${tgt_case}.txt
bpemodel=data/lang_1spm/${train_set}_${bpemode}${nbpe}_${tgt_case}
echo "dictionary: ${dict}"


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding Test"
    if [ ${dec_ngpu} = 1 ]; then
        nj=1
    fi

    pids=() # initialize pids
    if [ ${part} = 1 ]; then
        trans_set=$trans_set1
    elif [ ${part} = 2 ]; then
        trans_set=$trans_set2
    elif [ ${part} = 3 ]; then
        trans_set=$trans_set3
    fi
    for x in ${trans_set}; do
    (
        decode_dir=decode_${x}_$(basename ${decode_config%.*})
        feat_trans_dir=${dumpdir}/${x}/delta${do_delta}
        mkdir -p ${expdir}/${decode_dir}/asr

        # split data
        splitjson.py --parts ${nj} ${feat_trans_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            st_trans.py \
            --config ${decode_config} \
            --ngpu ${dec_ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --trans-json ${feat_trans_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --asr-si-result-label ${expdir}/${decode_dir}/asr/data.JOB.json \
            --model ${expdir}/results/${trans_model}

        if [[ ${x} = *tst20* ]] || [[ ${x} = *dev20* ]]; then
            set=$(echo ${x} | cut -f 1 -d "." | cut -f 3 -d "_")
            local/score_bleu_reseg.sh --case ${tgt_case} --bpe ${nbpe} --bpemodel ${bpemodel}.model \
                --remove_nonverbal ${remove_nonverbal} \
                ${expdir}/${decode_dir} ${dict} ${iwslt_test_data} ${set}
        else
            score_bleu.sh --case ${tgt_case} --bpe ${nbpe} --bpemodel ${bpemodel}.model \
                --remove_nonverbal ${remove_nonverbal} \
                ${expdir}/${decode_dir} "de" ${dict}
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
