# English
# --inference_args "--beam_size 1 --ctc_weight 0.0"\
# datasets/LibriSpeech/dev

s2t_exp=exp/s2t_train_s2t_multitask-ctc_ebf27_conv2d8_size1024_raw_bpe50000
inference_s2t_model=valid.total_count.ave_5best.till40epoch.pth

for x in datasets/TEDLIUM3_filtered/dev; do
    ./run.sh --stage 12 --stop_stage 13 \
        --test_sets $x \
        --s2t_exp ${s2t_exp} \
        --inference_s2t_model ${inference_s2t_model} \
        --cleaner whisper_en --hyp_cleaner whisper_en \
        --inference_nj 4 \
        --inference_args "--beam_size 1 --data_path_and_name_and_type dump/raw/$x/text.prev,text_prev,text" \
        --inference_tag ASR_beam1_${inference_s2t_model}_withprev

    ./run.sh --stage 12 --stop_stage 13 \
        --test_sets $x \
        --s2t_exp ${s2t_exp} \
        --inference_s2t_model ${inference_s2t_model} \
        --cleaner whisper_en --hyp_cleaner whisper_en \
        --inference_nj 4 \
        --inference_args "--beam_size 1" \
        --inference_tag ASR_beam1_${inference_s2t_model}
done
