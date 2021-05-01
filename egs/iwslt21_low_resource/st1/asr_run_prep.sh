. ./cmd.sh
. ./path.sh


# general configuration
backend=pytorch # chainer or pytorch
stage=4        # start from -1 if you need to start from data download
stop_stage=100
ngpu=1          # number of gpus during training ("0" uses cpu, otherwise use gpu)
dec_ngpu=0      # number of gpus during decoding ("0" uses cpu, otherwise use gpu)
nj=8            # number of parallel jobs for decoding
debugmode=1
dumpdir=dump    # directory to dump full features
N=0             # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0       # verbose option
resume=         # Resume the training from snapshot
seed=1          # seed to generate random number
# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/asr_train.yaml
decode_config=conf/decode.yaml

tag=""

# bpemode (unigram or bpe)
nbpe=2000
bpemode=bpe

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

src_lang=swa
tgt_lang=fr
src_case=lc.rm
tgt_case=lc.rm
dataset="sw-${tgt_lang}"

train_set=train_combine_sp
train_dev=valid.${src_lang}-${tgt_lang}.${tgt_lang}
recog_set="valid_org.${src_lang}-${tgt_lang}.${tgt_lang}"

feat_tr_dir=dump/${train_set}/deltafalse; mkdir -p ${feat_tr_dir}
feat_dt_dir=dump/${train_dev}/deltafalse; mkdir -p ${feat_dt_dir}

dict=data/${dataset}-all/bpe${nbpe}_units.txt
nlsyms=data/${dataset}-all/non_lang_syms.txt
bpemodel=data/${dataset}-all/bpe${nbpe}

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"

    echo "make json files"
    data2json.sh --nj 16 --feat ${feat_tr_dir}/feats.scp --text data/${train_set}/text.${tgt_case} --bpecode ${bpemodel}.model --lang ${tgt_lang} \
        data/${train_set} ${dict} > ${feat_tr_dir}/data_bpe${nbpe}.${src_case}_${tgt_case}.json

fi

if [ -z ${tag} ]; then
    expname=${train_set}_${tgt_case}_${backend}_$(basename ${train_config%.*})_${bpemode}${nbpe}
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${tgt_case}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --seed ${seed} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json
fi
