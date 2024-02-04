. ./path.sh

logdirs=exp/espnet/owsm_v3.1_ebf_base/st_covost2_beam1_valid.total_count.ave_5best.pth/test/CoVoST-2/test.en-*/logdir

for _logdir in ${logdirs}; do
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

grep "Total decoding time:" ${logdirs}/decoding_time.log
