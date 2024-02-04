set -e
set -u
set -o pipefail


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

tgt_case=lc.rm
s2t_exp=exp/s2t_train_s2t_multitask-ctc_ebf27_conv2d8_size1024_raw_bpe50000
inference_s2t_model=16epoch.pth
beam_size=1
inference_tag=st_covost2_beam${beam_size}_${inference_s2t_model}

# for tgt in de ca zh-CN fa et mn tr ar sv-SE lv sl ta ja id cy; do
for tgt in ca zh-CN; do
    case ${tgt} in
        "cy")
            download_id=cym ;;
        "id")
            download_id=ind ;;
        "ta")
            download_id=tam ;;
        "sl")
            download_id=slv ;;
        "lv")
            download_id=lav ;;
        "sv-SE")
            download_id=swe ;;
        "ar")
            download_id=ara ;;
        "tr")
            download_id=tur ;;
        "mn")
            download_id=mon ;;
        "et")
            download_id=est ;;
        "fa")
            download_id=fas ;;
        "ca")
            download_id=cat ;;
        "es")
            download_id=spa ;;
        "en")
            download_id=eng ;;
        "fr")
            download_id=fra ;;
        "nl")
            download_id=nld ;;
        "it")
            download_id=ita ;;
        "pt")
            download_id=por ;;
        "pl")
            download_id=pol ;;
        "de")
            download_id=deu ;;
        "ja")
            download_id=jpn ;;
        "zh-CN")
            download_id=zho ;;
    esac

    ./run.sh --stage 12 --stop_stage 12 \
        --test_sets "test/CoVoST-2/test.en-${tgt}" \
        --inference_s2t_model ${inference_s2t_model} \
        --s2t_exp ${s2t_exp} \
        --cleaner none --hyp_cleaner none \
        --inference_nj 8 \
        --inference_args "--beam_size ${beam_size} --lang_sym eng --task_sym st_${download_id}" \
        --inference_tag ${inference_tag}

    ./score_st.sh \
        --tgt_case ${tgt_case} \
        --src_lang en \
        --tgt_lang $(echo ${tgt} | cut -d'-' -f1) \
        --test_sets "test/CoVoST-2/test.en-${tgt}" \
        --s2t_exp ${s2t_exp} \
        --inference_tag ${inference_tag}

    _fs=16000
    _sample_shift=$(python3 -c "print(1 / ${_fs} * 1000)") # in ms
    _logdir=${s2t_exp}/${inference_tag}/test/CoVoST-2/test.en-${tgt}/logdir
    python3 pyscripts/utils/calculate_rtf.py \
        --log-dir ${_logdir} \
        --log-name "s2t_inference" \
        --input-shift ${_sample_shift} \
        --start-times-marker "speech length" \
        --end-times-marker "best hypo" \
        --inf-num 1 > ${_logdir}/decoding_time.log
done
