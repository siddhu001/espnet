#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=2        # start from -1 if you need to start from data download
stop_stage=100
nj=8            # number of parallel jobs for decoding
dumpdir=dump    # directory to dump full features
verbose=0       # verbose option
ngpu=8
nj=8            # number of parallel jobs for decoding
debugmode=1
seed=1
N=0
resume=
# bpemode (unigram or bpe)
nbpe=2000

. utils/parse_options.sh || exit 1;

train_config=conf/tuning/md/mdtrain-iwslt21-transformer.yaml
#train_config=conf/tuning/md/mdtrain-speechattn-small2.yaml
decode_config=conf/decode.yaml

# decoding parameter
trans_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=5                  # the number of ST models to be averaged
use_valbest_average=false     # if true, the validation `n_average`-best ST models will be averaged.
                             # if false, the last `n_average` ST models will be averaged.
metric=bleu                  # loss/acc/bleu

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

dataset="sw-en"
src_lang=swa
tgt_lang=en
src_case=lc.rm
tgt_case=lc.rm

train_set=train_sp.${src_lang}-${tgt_lang}.${tgt_lang}
train_dev=valid.${src_lang}-${tgt_lang}.${tgt_lang}
trans_set="valid_org.${src_lang}-${tgt_lang}.${tgt_lang}"

tag=joint_train

dict=data/${dataset}-all/bpe${nbpe}_units.txt
nlsyms=data/${dataset}-all/non_lang_syms.txt
bpemodel=data/${dataset}-all/bpe${nbpe}


feat_tr_dir=dump/${train_set}/deltafalse; mkdir -p ${feat_tr_dir}
feat_dt_dir=dump/${train_dev}/deltafalse; mkdir -p ${feat_dt_dir}

if [ -z ${tag} ]; then
    expname=${train_set}_${tgt_case}_${backend}_$(basename ${train_config%.*})_${bpemode}${nbpe}
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
    if [ -n "${asr_model}" ]; then
        expname=${expname}_asrtrans
    fi
    if [ -n "${mt_model}" ]; then
        expname=${expname}_mttrans
    fi
else
    expname=${train_set}_${tgt_case}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        st_train.py \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --n-iter-process 16 \
        --outdir ${expdir}/results \
	--save-interval-iters 8000 \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --seed ${seed} \
        --verbose ${verbose} \
        --resume ${resume} \
	--num-save-attention 0 \
	--num-save-ctc 0 \
        --train-json ${feat_tr_dir}/data_bpe${nbpe}.${src_case}_${tgt_case}.json \
	--train-asr-json dump/train_combine_sp/deltafalse/data_bpe${nbpe}.${src_case}_${tgt_case}.json \
	--train-mt-json dump/mt/data_bpe${nbpe}.${src_case}_${tgt_lang}.json   \
	--valid-json ${feat_dt_dir}/data_bpe${nbpe}.${src_case}_${tgt_case}.json
fi



if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
       [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]]; then
        # Average ST models
        if ${use_valbest_average}; then
            trans_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log --metric ${metric}"
        else
            trans_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${trans_model} \
            --num ${n_average}
    fi

    if [ ${dec_ngpu} = 1 ]; then
        nj=1
    fi

    pids=() # initialize pids
    for x in ${trans_set}; do
    (
        decode_dir=decode_${x}_$(basename ${decode_config%.*})
        feat_trans_dir=${dumpdir}/${x}/delta${do_delta}

        # reset log for RTF calculation
        if [ -f ${expdir}/${decode_dir}/log/decode.1.log ]; then
            rm ${expdir}/${decode_dir}/log/decode.*.log
        fi

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
            --model ${expdir}/results/${trans_model}

        score_bleu.sh --case ${tgt_case} --bpe ${nbpe} --bpemodel ${bpemodel}.model \
            ${expdir}/${decode_dir} ${tgt_lang} ${dict}

        calculate_rtf.py --log-dir ${expdir}/${decode_dir}/log
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
