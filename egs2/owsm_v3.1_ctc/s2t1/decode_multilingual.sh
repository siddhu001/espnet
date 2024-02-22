s2t_exp=exp/s2t_train_s2t_multitask-ctc_ebf27_conv2d8_size1024_raw_bpe50000
inference_s2t_model=valid.total_count.ave_5best.till40epoch.pth
beam_size=1
inference_tag=ASR_beam${beam_size}_${inference_s2t_model}


# MLS
# for lang in it pt pl es fr de nl; do
for lang in es fr de nl; do
    case ${lang} in
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
    esac

    ./run.sh --stage 12 --stop_stage 13 \
        --test_sets "test/MLS/${lang}_test" \
        --s2t_exp ${s2t_exp} \
        --inference_s2t_model ${inference_s2t_model} \
        --cleaner whisper_basic --hyp_cleaner whisper_basic \
        --inference_nj 4 \
        --inference_args "--beam_size ${beam_size} --lang_sym ${download_id}" \
        --inference_tag ${inference_tag}
done


# Chinese
./run.sh --stage 12 --stop_stage 13 \
    --test_sets "test/AISHELL-1/test" \
    --inference_s2t_model ${inference_s2t_model} \
    --s2t_exp ${s2t_exp} \
    --cleaner whisper_basic --hyp_cleaner whisper_basic \
    --inference_nj 8 \
    --nlsyms_txt data/nlsyms_scoring.txt \
    --inference_args "--beam_size ${beam_size} --lang_sym zho" \
    --inference_tag ${inference_tag}


# Japanese
./run.sh --stage 12 --stop_stage 13 \
    --test_sets "test/ReazonSpeech/test" \
    --inference_s2t_model ${inference_s2t_model} \
    --s2t_exp ${s2t_exp} \
    --cleaner whisper_basic --hyp_cleaner whisper_basic \
    --inference_nj 8 \
    --nlsyms_txt data/nlsyms_scoring.txt \
    --inference_args "--beam_size ${beam_size} --lang_sym jpn" \
    --inference_tag ${inference_tag}


# Korean
./run.sh --stage 12 --stop_stage 13 \
    --test_sets "test/KsponSpeech/eval_clean test/KsponSpeech/eval_other" \
    --inference_s2t_model ${inference_s2t_model} \
    --s2t_exp ${s2t_exp} \
    --cleaner whisper_basic --hyp_cleaner whisper_basic \
    --inference_nj 8 \
    --nlsyms_txt data/nlsyms_scoring.txt \
    --inference_args "--beam_size ${beam_size} --lang_sym kor" \
    --inference_tag ${inference_tag}
