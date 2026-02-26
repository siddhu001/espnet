import json
import os

conv_id_done={}
with open("speech_response_dpo.txt", 'w') as out:
    for i in range(1,4):
        for j in range(1):
            file_path="exp/speechlm_audio_dialogue_fisher_train_delay_olmo2_7b_dpo_dialogue_duplex_last/decode_general_limit_tts_duplex_3epoch/audio_dialogue_eval2000_duplex_cot_no_asr2_response_last/log/output."+str(i)+"/dialogue/rank"+str(j)+"_dialogue.json"
            if os.path.exists(file_path):
                dict1=json.load(open(file_path))
                for data in dict1:
                    # Get the conversation ID (first key)
                    conv_id = list(data.keys())[0]
                    if conv_id not in conv_id_done:
                        conv_id_done[conv_id]=1
                    else:
                        continue

                    # Get the last turn's text (index 3)
                    last_text = data[conv_id][-1][3]

                    out.write(conv_id+" "+last_text+"\n")
    for i in range(5,9):
        for j in range(1):
            file_path="exp/speechlm_audio_dialogue_fisher_train_delay_olmo2_7b_dpo_dialogue_duplex_last/decode_general_limit_tts_duplex_3epoch/audio_dialogue_eval2000_duplex_cot_no_asr2_response_last/log/output."+str(i)+"/dialogue/rank"+str(j)+"_dialogue.json"
            if os.path.exists(file_path):
                dict1=json.load(open(file_path))
                for data in dict1:
                    # Get the conversation ID (first key)
                    conv_id = list(data.keys())[0]
                    if conv_id not in conv_id_done:
                        conv_id_done[conv_id]=1
                    else:
                        continue

                    # Get the last turn's text (index 3)
                    last_text = data[conv_id][-1][3]

                    out.write(conv_id+" "+last_text+"\n")
    # for i in range(5,9):
    #     for j in range(1):
    #         dict1=json.load(open("exp/speechlm_audio_dialogue_fisher_train_delay_olmo2_7b_dpo_dialogue_copy/decode_general_limit_tts2_2epoch/audio_dialogue_eval2000_text_response_dpo_correct_2epoch/log/output."+str(i)+"/dialogue/rank"+str(j)+"_dialogue.json"))
    #         for data in dict1:
    #             # Get the conversation ID (first key)
    #             conv_id = list(data.keys())[0]

    #             # Get the last turn's text (index 3)
    #             last_text = data[conv_id][-1][3]

    #             out.write(conv_id+" "+last_text+"\n")

# ./scripts/utils/evaluate_asr.sh \
        #     --whisper_tag ${whisper_tag} \
        #     --whisper_dir ${whisper_dir} \
        #     --cleaner ${cleaner} \
        #     --hyp_cleaner ${hyp_cleaner} \
        #     --inference_nj ${inference_nj} \
        #     --nj ${nj} \
        #     --gt_text ${ref_dir}/text \
        #     --gpu_inference ${gpu_inference} \
        #     ${gen_dir}/dialogue ${gen_dir}/scoring/eval_wer