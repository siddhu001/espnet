stage=1
stop_stage=10000

data=../asr1/data
dsets="train valid test"

confname="bart_large_v7"

asrtext=whisper_ASR_updated_transcript/text
asrtag="whftv2"

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
    python local/add_asr_result.py ${asrtext} --tag ${asrtag}

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
    bash score.sh exp/${confname}/output.txt
    
    echo "Stage 4-2: Scoring on ASR"
    bash score.sh exp/${confname}/output_asr${asrtag}.txt
fi
