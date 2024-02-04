# ./run.sh \
#     --s2t_exp exp/s2t_train_s2t_finetune_asronly_ebf_e24_lr2e-4_raw_bpe5000 \
#     --stage 12 \
#     --stop_stage 13 \
#     --gpu_inference true \
#     --inference_nj 4 \
#     --inference_s2t_model valid.loss_ctc.ave_10best.pth \
#     --cleaner whisper_en --hyp_cleaner whisper_en


./run.sh \
    --s2t_exp exp/s2t_train_s2t_finetune_stonly_ebf_e24_lr2e-4_raw_bpe5000 \
    --stage 12 \
    --stop_stage 12 \
    --gpu_inference true \
    --inference_nj 8 \
    --inference_s2t_model valid.total_count.ave_10best.pth \
    --inference_args "--task_sym st_de"
