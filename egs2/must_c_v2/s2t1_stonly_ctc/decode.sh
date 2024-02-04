# valid.total_count.ave_10best.pth valid.loss_ctc.ave_10best.pth 
        # --test_sets "dev.en-de"

for model in valid.cer_ctc.ave_10best.pth; do
    ./run.sh \
        --s2t_exp exp/s2t_train_s2t_scratch_stonly_ebf_e24_lr1e-3_warmup25k_raw_bpe10000 \
        --stage 12 \
        --stop_stage 12 \
        --gpu_inference true \
        --inference_nj 8 \
        --inference_s2t_model ${model} \
        --inference_args "--task_sym st_de"
done
