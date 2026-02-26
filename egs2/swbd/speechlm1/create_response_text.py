import json

with open("text_response_dpo.txt", 'w') as out:
    for i in range(1,9):
        for j in range(6):
            dict1=json.load(open("/work/nvme/bbjs/arora1/speech_lm/delta_ai/newer_branch/espnet/egs2/swbd/speechlm1/exp/speechlm_audio_dialogue_fisher_train_delay_olmo2_7b_dpo_dialogue_duplex_last/decode_topk_duplex_response_2epoch/audio_dialogue_eval2000_duplex_cot_no_asr2/log/output."+str(i)+"/dialogue/rank"+str(j)+"_dialogue.json"))
            for data in dict1:
                # Get the conversation ID (first key)
                conv_id = list(data.keys())[0]

                # Get the last turn's text (index 3)
                last_text = data[conv_id][-1][3]

                out.write(conv_id+" "+last_text+"\n")