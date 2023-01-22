stage=4
stop_stage=4
#stop_stage=10000

data=../asr1/data
dsets="train valid test"

confname="bart_large_v7"

asrdir=../asr1_pipeline/exp/asr_train_asr2_wavlm_mylibricvpt_raw_en_bpe500_sp/decode_asr_beam20_ctc0.5_lm_weight0.5_lm_lm_train_rnn_lm_en_bpe500_valid.loss.best_asr_model_valid.acc.ave_10best/test/

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Prepare data"
    for dset in ${dsets}; do
        mkdir -p data/${dset}
        # copy input
        cp ${data}/${dset}/transcript data/${dset}/.
        # copy output
        cp ${data}/${dset}/text data/${dset}/.

        # create json
        python local/prepare_data.py data/${dset}
    done

    echo "Prepare ASR results"
    python local/add_asr_result.py ${asrdir}

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: NLU training"
    python run.py conf/${confname}.yaml
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3-1: Decoding on GT"
    python run.py conf/pred_${confname}.yaml
    echo "Stage 3-2: Decoding on ASR"
    python run.py conf/pred_asr_${confname}.yaml
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4-1: Scoring on GT"
    python local/score_sclite.py exp/${confname}/output_best.txt

    $PWD/../../../tools/sctk-2.4.10/bin/sclite \
        -r "exp/${confname}/output_best/ref.trn" trn \
        -h "exp/${confname}/output_best/hyp.trn" trn \
        -i rm -o all stdout > "exp/${confname}/output_best/result.txt"
    
    grep -e Avg -e SPKR -m 2 "exp/${confname}/output_best/result.txt"
    
    echo "Stage 4-2: Scoring on ASR"
    python local/score_sclite.py exp/${confname}/output_best_asr.txt

    $PWD/../../../tools/sctk-2.4.10/bin/sclite \
        -r "exp/${confname}/output_best_asr/ref.trn" trn \
        -h "exp/${confname}/output_best_asr/hyp.trn" trn \
        -i rm -o all stdout > "exp/${confname}/output_best_asr/result.txt"
    
    grep -e Avg -e SPKR -m 2 "exp/${confname}/output_best_asr/result.txt"
fi
