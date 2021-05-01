. ./cmd.sh 
. ./path.sh

# bpemode (unigram or bpe)
nbpe=2000

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=0
stop_stage=100
src_lang=swa
tgt_lang=en
src_case=lc.rm
tgt_case=lc.rm
dataset="sw-${tgt_lang}"

train_set=train_sp.${src_lang}-${tgt_lang}.${tgt_lang}
train_dev=valid.${src_lang}-${tgt_lang}.${tgt_lang}
trans_set="valid_org.${src_lang}-${tgt_lang}.${tgt_lang}"

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
    for x in ${train_dev} ${trans_set}; do
        feat_trans_dir=dump/${x}/deltafalse
        data2json.sh --feat ${feat_trans_dir}/feats.scp --text data/${x}/text.${tgt_case} --bpecode ${bpemodel}.model --lang ${tgt_lang} \
            data/${x} ${dict} > ${feat_trans_dir}/data_bpe${nbpe}.${src_case}_${tgt_case}.json
    done

    # update json (add source references)
    for x in ${train_set} ${train_dev} ${trans_set}; do
        feat_dir=dump/${x}/deltafalse
        data_dir=data/$(echo ${x} | cut -f 1 -d ".").${src_lang}-${tgt_lang}.${src_lang}
        update_json.sh --text ${data_dir}/text.${src_case} --bpecode ${bpemodel}.model \
            ${feat_dir}/data_bpe${nbpe}.${src_case}_${tgt_case}.json ${data_dir} ${dict}
    done
fi
