# English
# "test/WSJ/test_eval92 test/LibriSpeech/test_clean test/LibriSpeech/test_other test/TEDLIUM/test test/SWBD/eval2000 test/CommonVoice/eng_test test/FLEURS/test_eng test/VoxPopuli/test_eng"
# --inference_args "--beam_size 1 --ctc_weight 0.0"

s2t_exp=exp/s2t_train_s2t_multitask-ctc_ebf27_conv2d8_size1024_raw_bpe50000
inference_s2t_model=16epoch.pth

./run.sh --stage 12 --stop_stage 13 \
    --test_sets "test/WSJ/test_eval92 test/LibriSpeech/test_clean test/LibriSpeech/test_other test/TEDLIUM/test test/SWBD/eval2000 test/CommonVoice/eng_test test/FLEURS/test_eng test/VoxPopuli/test_eng" \
    --inference_s2t_model ${inference_s2t_model} \
    --s2t_exp ${s2t_exp} \
    --cleaner whisper_en --hyp_cleaner whisper_en \
    --inference_nj 8 \
    --inference_args "--beam_size 1"
