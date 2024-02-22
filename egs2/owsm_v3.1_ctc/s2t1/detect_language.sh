s2t_exp=exp/s2t_train_s2t_multitask-ctc_ebf27_conv2d8_size1024_raw_bpe50000
inference_s2t_model=valid.total_count.ave_5best.till40epoch.pth

./run.sh --stage 12 --stop_stage 12 \
    --test_sets "test/FLEURS/test" \
    --inference_s2t_model ${inference_s2t_model} \
    --s2t_exp ${s2t_exp} \
    --inference_nj 4 \
    --inference_args "--beam_size 1 --lang_sym nolang" \
    --inference_tag LID_${inference_s2t_model}
