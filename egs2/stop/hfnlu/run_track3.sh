stage=1
stop_stage=10000

lrsplits=../asr1/db/low_resource_splits/

confname_all="bart_large_tr3"
confname_r="bart_large_tr3r_v2"
confname_w="bart_large_tr3w_v2"
confname_r_aug="bart_large_tr3r_v2_aug10x1m0.2th0"
confname_w_aug="bart_large_tr3w_v2_aug10x1m0.2th0"

asrtext_r=whisper_FT_dup20_lower_iter1k_trans/text_reminder
asrtext_w=whisper_FT_dup20_lower_iter1k_trans/text_weather
asrtag="whftfd20lowit1k"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Prepare data"
    # assume Track1 data is already processed
    
    python local/split_track3_data.py ${lrsplits}
    
    echo "Prepare ASR results"
    python local/add_asr_result.py ${asrtext_r} --dset test-reminder --tag whftbest
    python local/add_asr_result.py ${asrtext_w} --dset test-weather --tag whftbest
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: NLU training on all Track3 data"
    python run.py conf/${confname_all}.yaml

    # set the trained model for Stage 3
    python local/set_ptmodel.py exp/${confname_all}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3-1: Fine-tune Stage 2 NLU on reminder"
    python run.py conf/${confname_r}.yaml
    echo "Stage 3-2: Fine-tune Stage 2 NLU on weather"
    python run.py conf/${confname_w}.yaml
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4-1: Decoding on GT"
    python run.py conf/pred_${confname_r}.yaml
    python run.py conf/pred_${confname_w}.yaml
fi

### Data augmentation

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Stage 5: Data augmentation"
    python local/aug_data_bert.py data/train-reminder-25spis/data.json --mask 0.2
    python local/aug_data_bert.py data/train-weather-25spis/data.json --mask 0.2

    echo "Decoding for filtering"
    python run.py conf/flt_aug_${confname_r}.yaml data/train-reminder-25spis/data_aug10x1m0.20.json exp/${confname_r}/output_flt.txt
    python run.py conf/flt_aug_${confname_w}.yaml data/train-weather-25spis/data_aug10x1m0.20.json exp/${confname_w}/output_flt.txt

    # generate `data/train-{reminder/weather}-25spis/data_aug10x1m0.20_th0.00.json`
    python local/flt_aug_data.py data/train-reminder-25spis/data_aug10x1m0.20.json --th 0
    python local/flt_aug_data.py data/train-weather-25spis/data_aug10x1m0.20.json --th 0
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "Stage 6-1: Fine-tune Stage 2 NLU on reminder (augmentated)"
    python run.py conf/${confname_r_aug}.yaml
    echo "Stage 6-2: Fine-tune Stage 2 NLU on weather (augmentated)"
    python run.py conf/${confname_w_aug}.yaml
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "Stage 7: Decoding on GT"
    python run.py conf/pred_${confname_r_aug}.yaml
    python run.py conf/pred_${confname_w_aug}.yaml
fi

### Retrieval augmentation

confname_retro_all="bart_large_tr3_retro_v2"
confname_retro_r_aug="bart_large_tr3r_retro_aug10x1m0.2th0_v2_fix"
confname_retro_w_aug="bart_large_tr3w_retro_aug10x1m0.2th0_v2_fix"

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "Stage 8: Prepare data for retrieval augmentation"

    python retro/build_embeds.py
    # Prepare `data/train-tr3/data_retro__p0.10_cat4_x30.json`
    python retro/prepare_data_exams.py --times 30
    # Prepare `data/valid-tr3/data_retro__cat4.json`
    python retro/prepare_data.py data/valid-tr3/data.json
    python retro/prepare_data.py data/valid-reminder-25spis/data.json
    python retro/prepare_data.py data/valid-weather-25spis/data.json
    # Prepare `data/test-{reminder/weather}/data_retro__cat4.json`
    python retro/prepare_data.py data/test-reminder/data.json
    python retro/prepare_data.py data/test-weather/data.json
    
    # Prepare `data/train-{reminder/weather}-25spis/data_aug10x1m0.20_th0.00_retro__p0.10_cat4_x10.json`
    python retro/prepare_data.py data/train-reminder-25spis/data_aug10x1m0.20_th0.00.json --train --times 10
    python retro/prepare_data.py data/train-weather-25spis/data_aug10x1m0.20_th0.00.json --train --times 10
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "Stage 9: Retrieval NLU training on all Track3 data"
    python run.py conf/${confname_retro_all}.yaml

    # set the trained model for Stage 9
    python local/set_ptmodel.py exp/${confname_retro_all}
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "Stage 10-1: Fine-tune Stage 9 NLU on reminder (aug)"
    python run.py conf/${confname_retro_r_aug}.yaml
    echo "Stage 10-2: Fine-tune Stage 9 NLU on weather (aug)"
    python run.py conf/${confname_retro_w_aug}.yaml
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    # Add examplers to ASR results
    python retro/prepare_data.py data/test-reminder/data_asr${asrtag}.json
    python retro/prepare_data.py data/test-weather/data_asr${asrtag}.json

    echo "Stage 11-1: Decoding on GT"
    python run.py conf/pred_${confname_retro_r_aug}.yaml
    python run.py conf/pred_${confname_retro_w_aug}.yaml
    echo "Stage 11-2: Decoding on ASR"
    python run.py conf/pred_asr_${confname_retro_r_aug}.yaml
    python run.py conf/pred_asr_${confname_retro_w_aug}.yaml
fi
