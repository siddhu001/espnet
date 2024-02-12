./run.sh --slu_config conf//train_asr_hubert.yaml --pretrained_model /scratch/bbjs/arora1/espnet_slue_PR/espnet/egs2/tedlium3/asr1/exp/asr_train_asr_hubert_raw_en_bpe500/valid.acc.ave_10best.pth:::ctc --inference_config conf/decode_asr.yaml --train_set train --valid_set devel --test_sets 'test devel' --lm_train_text data/train/text --bpe_train_text data/train/text --stage 11 --ngpu 4 --stop_stage 12 --inference_nj 1 --inference_slu_model valid.acc.ave.pth
./run.sh --slu_config conf//train_asr_slurp_weighted.yaml --pretrained_model /scratch/bbjs/arora1/espnet_slue_PR/espnet/egs2/tedlium3/asr1/exp/asr_train_asr_slurp_weighted_raw_en_bpe500/valid.acc.ave_10best.pth:::ctc --inference_config conf/decode_asr.yaml --train_set train --valid_set devel --test_sets 'test devel' --lm_train_text data/train/text --bpe_train_text data/train/text --stage 11 --ngpu 4 --stop_stage 12 --inference_nj 1 --inference_slu_model valid.acc.ave.pth