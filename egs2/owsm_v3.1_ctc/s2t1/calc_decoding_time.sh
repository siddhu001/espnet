. ./path.sh

# logdirs=exp/espnet/owsm_v3.1_ebf_base/st_covost2_beam1_valid.total_count.ave_5best.pth/test/CoVoST-2/test.en-*/logdir

# for _logdir in ${logdirs}; do
#     _fs=16000
#     _sample_shift=$(python3 -c "print(1 / ${_fs} * 1000)") # in ms

#     python3 pyscripts/utils/calculate_rtf.py \
#         --log-dir ${_logdir} \
#         --log-name "s2t_inference" \
#         --input-shift ${_sample_shift} \
#         --start-times-marker "speech length" \
#         --end-times-marker "best hypo" \
#         --inf-num 1 > ${_logdir}/decoding_time.log
# done

# grep "Total decoding time:" ${logdirs}/decoding_time.log


test_sets="test/WSJ/test_eval92 test/LibriSpeech/test_clean test/LibriSpeech/test_other test/SWBD/eval2000 test/TEDLIUM/test test/VoxPopuli/test_eng test/CommonVoice/eng_test test/FLEURS/test_eng test/MLS/en_test"
output_dir=exp/s2t_train_s2t_multitask-ctc_ebf27_conv2d8_size1024_raw_bpe50000/ASR_beam1_valid.total_count.ave_5best.till40epoch.pth

for x in ${test_sets}; do
    _logdir=${output_dir}/${x}/logdir
    _fs=16000
    _sample_shift=$(python3 -c "print(1 / ${_fs} * 1000)") # in ms

    python3 pyscripts/utils/calculate_rtf.py \
        --log-dir ${_logdir} \
        --log-name "s2t_inference" \
        --input-shift ${_sample_shift} \
        --start-times-marker "speech length" \
        --end-times-marker "best hypo" \
        --inf-num 1 > ${_logdir}/decoding_time.log
done

for x in ${test_sets}; do
    _logdir=${output_dir}/${x}/logdir
    echo ${_logdir}
    grep "Total decoding time:" ${_logdir}/decoding_time.log
done
