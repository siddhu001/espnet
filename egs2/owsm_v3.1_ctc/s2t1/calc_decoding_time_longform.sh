. ./path.sh

logdirs=exp/s2t_train_s2t_multitask-ctc_ebf27_conv2d8_size1024_raw_bpe50000/ASR_TEDLIUM-long_context*_batch32_valid.total_count.ave_5best.till40epoch.pth

for _logdir in ${logdirs}; do
    _fs=16000
    _sample_shift=$(python3 -c "print(1 / ${_fs} * 1000)") # in ms

    python3 pyscripts/utils/calculate_rtf.py \
        --log-dir ${_logdir} \
        --log-name "decode" \
        --input-shift ${_sample_shift} \
        --start-times-marker "speech length" \
        --end-times-marker "best hypo" \
        --inf-num 1 > ${_logdir}/decoding_time.log
done

grep "Total decoding time:" ${logdirs}/decoding_time.log
