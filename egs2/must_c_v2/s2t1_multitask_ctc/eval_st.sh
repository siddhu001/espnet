set -e
set -u
set -o pipefail


tgt_case=tc
tgt_lang=de
test_sets="tst-COMMON.en-de tst-HE.en-de"
data_feats=dump/raw

cleaner=none
hyp_cleaner=$cleaner
python=python3
nlsyms_txt=none

st_exp=exp/s2t_train_s2t_multitask_ebf24_lr1e-3_warmup25k_conv2d8_asrctc6-12-18_raw_bpe10000
inference_tag=decode_s2t_task_symst_de_s2t_model_valid.total_count.ave_5best


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "Stage 13: Scoring"

for dset in ${test_sets}; do
    _data="${data_feats}/${dset}"
    _dir="${st_exp}/${inference_tag}/${dset}"

    # TODO(jiatong): add asr scoring and inference

    _scoredir="${_dir}/score_bleu"
    mkdir -p "${_scoredir}"

    paste \
        <(<"${_data}/text.${tgt_case}.${tgt_lang}" \
            ${python} -m espnet2.bin.tokenize_text  \
                -f 2- --input - --output - \
                --token_type word \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --remove_non_linguistic_symbols true \
                --cleaner "${cleaner}" \
                ) \
        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
            >"${_scoredir}/ref.trn.org"

    paste \
        <(<"${_dir}/text_nospecial"  \
                ${python} -m espnet2.bin.tokenize_text  \
                    -f 2- --input - --output - \
                    --token_type word \
                    --non_linguistic_symbols "${nlsyms_txt}" \
                    --remove_non_linguistic_symbols true \
                    --cleaner "${hyp_cleaner}" \
                    ) \
        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
            >"${_scoredir}/hyp.trn.org"

    # remove utterance id
    perl -pe 's/\([^\)]+\)$//g;' "${_scoredir}/ref.trn.org" > "${_scoredir}/ref.trn"
    perl -pe 's/\([^\)]+\)$//g;' "${_scoredir}/hyp.trn.org" > "${_scoredir}/hyp.trn"

    # detokenizer
    detokenizer.perl -l ${tgt_lang} -q < "${_scoredir}/ref.trn" > "${_scoredir}/ref.trn.detok"
    detokenizer.perl -l ${tgt_lang} -q < "${_scoredir}/hyp.trn" > "${_scoredir}/hyp.trn.detok"

    # rotate result files
    if [ ${tgt_case} = "tc" ]; then
        pyscripts/utils/rotate_logfile.py ${_scoredir}/result.tc.txt
    fi
    pyscripts/utils/rotate_logfile.py ${_scoredir}/result.lc.txt

    if [ ${tgt_case} = "tc" ]; then
        echo "Case sensitive BLEU result (single-reference)" > ${_scoredir}/result.tc.txt
        sacrebleu "${_scoredir}/ref.trn.detok" \
                    -i "${_scoredir}/hyp.trn.detok" \
                    -m bleu chrf ter \
                    >> ${_scoredir}/result.tc.txt

        log "Write a case-sensitive BLEU (single-reference) result in ${_scoredir}/result.tc.txt"
    fi

    # detokenize & remove punctuation except apostrophe
    scripts/utils/remove_punctuation.pl < "${_scoredir}/ref.trn.detok" > "${_scoredir}/ref.trn.detok.lc.rm"
    scripts/utils/remove_punctuation.pl < "${_scoredir}/hyp.trn.detok" > "${_scoredir}/hyp.trn.detok.lc.rm"
    echo "Case insensitive BLEU result (single-reference)" > ${_scoredir}/result.lc.txt
    sacrebleu -lc "${_scoredir}/ref.trn.detok.lc.rm" \
                -i "${_scoredir}/hyp.trn.detok.lc.rm" \
                -m bleu chrf ter \
                >> ${_scoredir}/result.lc.txt
    log "Write a case-insensitve BLEU (single-reference) result in ${_scoredir}/result.lc.txt"

done

# Show results in Markdown syntax
scripts/utils/show_translation_result.sh --case $tgt_case "${st_exp}" > "${st_exp}"/RESULTS.md
cat "${st_exp}"/RESULTS.md
