        # --test_sets "dev.en-de"
# valid.loss_ctc.ave_10best.pth valid.cer_ctc.ave_10best.pth

s2t_exp=exp/s2t_train_s2t_multitask_ebf24_lr1e-3_warmup25k_conv2d8_asrctc6-12-18_raw_bpe10000

for model in valid.total_count.ave_5best.pth; do
    ./run.sh \
        --s2t_exp $s2t_exp \
        --stage 12 \
        --stop_stage 13 \
        --gpu_inference true \
        --inference_nj 4 \
        --inference_s2t_model ${model} \
        --cleaner none --hyp_cleaner none

    ./run.sh \
        --s2t_exp $s2t_exp \
        --stage 12 \
        --stop_stage 12 \
        --gpu_inference true \
        --inference_nj 4 \
        --inference_s2t_model ${model} \
        --inference_args "--task_sym st_de"
done
