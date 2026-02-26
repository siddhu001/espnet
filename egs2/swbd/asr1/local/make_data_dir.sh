# python local/data_prep_swbd_s2s_history_dpo_duplex.py dump/raw_audio_dialogue_fisher/train_dpo_duplex_cot/data/dialogue_rank3.json dump/raw_codec_ssl_tts_librispeech_100/train_nodup_dpo/wav.scp dump/raw_audio_dialogue_fisher/train_dpo_duplex_final2/ dump/raw_audio_dialogue_fisher/valid_dpo_duplex_final2/
# dir=dump/raw_audio_dialogue_fisher/train_dpo_duplex_final2/
# cp ${dir}/data/dialogue.1 ${dir}/dialogue
# python3 pyscripts/utils/make_speechlm_json.py \
#   --task audio_dialogue \
#   --output_json ${dir}/data.json \
#   --file_modality_type ${dir}/dialogue,dialogue,dialogue_json

# dir=dump/raw_audio_dialogue_fisher/valid_dpo_duplex_final2/
# cp ${dir}/data/dialogue.1 ${dir}/dialogue
# python3 pyscripts/utils/make_speechlm_json.py \
#   --task audio_dialogue \
#   --output_json ${dir}/data.json \
#   --file_modality_type ${dir}/dialogue,dialogue,dialogue_json

# python local/data_prep_swbd_s2s_history_dpo_utmos.py dump/raw_audio_dialogue_fisher/train_nodup_response_dpo_emotion/data/dialogue_rank1.json  dump/raw_audio_dialogue_fisher/train_dpo_emotion_final/ dump/raw_audio_dialogue_fisher/valid_dpo_emotion_final/
# dir=dump/raw_audio_dialogue_fisher/train_dpo_emotion_final/
# cp ${dir}/data/dialogue.1 ${dir}/dialogue
# python3 pyscripts/utils/make_speechlm_json.py \
#   --task audio_dialogue \
#   --output_json ${dir}/data.json \
#   --file_modality_type ${dir}/dialogue,dialogue,dialogue_json

# dir=dump/raw_audio_dialogue_fisher/valid_dpo_emotion_final/
# cp ${dir}/data/dialogue.1 ${dir}/dialogue
# python3 pyscripts/utils/make_speechlm_json.py \
#   --task audio_dialogue \
#   --output_json ${dir}/data.json \
#   --file_modality_type ${dir}/dialogue,dialogue,dialogue_json


# python local/data_prep_swbd_audio_history_dpo_eval_final.py dump/raw_audio_dialogue_fisher/eval2000_response/data/dialogue_rank1.json dump/raw_audio_dialogue_fisher/eval2000_asr_combined_full_2epoch/ asr_audio_dpo_combined_full_3epoch_final.txt
# dir=dump/raw_audio_dialogue_fisher/eval2000_asr_combined_full_2epoch/
# cp ${dir}/data/dialogue.1 ${dir}/dialogue
# python3 pyscripts/utils/make_speechlm_json.py \
#   --task audio_dialogue \
#   --output_json ${dir}/data.json \
#   --file_modality_type ${dir}/dialogue,dialogue,dialogue_json

# python local/data_prep_swbd_audio_history_dpo_eval_res_final.py dump/raw_audio_dialogue_fisher/eval2000_asr_combined_full_2epoch/data/dialogue_rank1_pack1.json dump/raw_audio_dialogue_fisher/eval2000_text_response_dpo_combined_full_2epoch/ text_response_audio_dpo_combined_full_2epoch_no_halluc_short2.txt
# dir=dump/raw_audio_dialogue_fisher/eval2000_text_response_dpo_combined_full_2epoch/
# cp ${dir}/data/dialogue.1 ${dir}/dialogue
# python3 pyscripts/utils/make_speechlm_json.py \
#   --task audio_dialogue \
#   --output_json ${dir}/data.json \
#   --file_modality_type ${dir}/dialogue,dialogue,dialogue_json

# python local/data_prep_swbd_s2s_history_dpo_duplex_eval_new.py dump/raw_audio_dialogue_fisher/eval2000_duplex_cot_no_asr/data/dialogue_rank2.json dump/raw_audio_dialogue_fisher/eval2000_duplex_cot_no_asr_new/
# dir=dump/raw_audio_dialogue_fisher/eval2000_duplex_cot_no_asr_new/
# cp ${dir}/data/dialogue.1 ${dir}/dialogue
# python3 pyscripts/utils/make_speechlm_json.py \
#   --task audio_dialogue \
#   --output_json ${dir}/data.json \
#   --file_modality_type ${dir}/dialogue,dialogue,dialogue_json

python local/data_prep_swbd_audio_history_dpo_duplex_eval_res_final.py dump/raw_audio_dialogue_fisher/eval2000_duplex_cot_no_asr2_subset/data/dialogue_rank1_pack1.json dump/raw_audio_dialogue_fisher/eval2000_duplex_cot_no_asr2_response_last_topk_2epoch/ text_response_topk_dpo_duplex_last_2epoch_no_halluc.txt
dir=dump/raw_audio_dialogue_fisher/eval2000_duplex_cot_no_asr2_response_last_topk_2epoch/
cp ${dir}/data/dialogue.1 ${dir}/dialogue
python3 pyscripts/utils/make_speechlm_json.py \
  --task audio_dialogue \
  --output_json ${dir}/data.json \
  --file_modality_type ${dir}/dialogue,dialogue,dialogue_json
