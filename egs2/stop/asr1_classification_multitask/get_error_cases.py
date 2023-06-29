# from local.evaluation.util import load_gold_data, load_predictions
gold_text=open("/projects/bbjs/arora1/new_download/espnet/egs2/stop/asr3_combined/exp/asr_train_asr_whisper_full_correct_specaug_raw_en_whisper_multilingual/decode_asr_sp_asr_model_valid.acc.ave/test/score_wer/ref.trn")
pred_text=open("/projects/bbjs/arora1/new_download/espnet/egs2/stop/asr3_combined/exp/asr_train_asr_whisper_full_correct_specaug_raw_en_whisper_multilingual/decode_asr_sp_asr_model_valid.acc.ave/test/score_wer/hyp.trn")
pred_text_correct=open("/projects/bbjs/arora1/new_download/espnet/egs2/stop/asr2_combined/exp/asr_train_asr_whisper_full_correct_specaug_raw_en_whisper_multilingual/decode_asr_sp_asr_model_valid.acc.ave/test/score_wer/hyp.trn")
gold_arr=[line for line in gold_text]
pred_arr=[line for line in pred_text]
pred_correct_arr=[line for line in pred_text_correct]
for i in range(len(gold_arr)):
    if gold_arr[i]==pred_correct_arr[i]:
        if gold_arr[i]!=pred_arr[i]:
            if pred_arr[i].strip().count("[")==pred_arr[i].strip().count("]"):
                if pred_arr[i].strip().count("[")==gold_arr[i].strip().count("["):
                    pred_arr1=[k.split()[0] for k in pred_arr[i].split("sl:")[1:]]
                    gold_arr1=[k.split()[0] for k in gold_arr[i].split("sl:")[1:]]
                    if pred_arr1!=gold_arr1:
                        print(gold_arr[i].strip())
                        print(pred_arr[i].strip())
                        print("NEXT")
                        # exit()
                    # print(gold_arr[i].strip())
                    # print(pred_arr[i].strip())
                    # print("NEXT")
# gold_examples = load_gold_data("/scratch/bbjs/arora1/slurp/dataset/slurp/test.jsonl",False)
# pred_nostop = load_predictions("result_test_nostop.json",False)
# pred = load_predictions("exp/asr_train_asr_whisper_full_correct_raw_en_whisper_multilingual/result_test.json",False)
# total_entity_type={}
# error_type={}
# for k in gold_examples:
#     a1="{}_{}".format(gold_examples[k]["scenario"], gold_examples[k]["action"])
#     a1_pred="{}_{}".format(pred[k]["scenario"], pred[k]["action"])
#     a1_pred_nostop="{}_{}".format(pred_nostop[k]["scenario"], pred_nostop[k]["action"])
#     if a1==a1_pred_nostop:
#         if a1!=a1_pred:
#             if a1 not in error_type:
#                 error_type[a1]=0
#             error_type[a1]+=1

#             # print(gold_examples[k]["text"])
#             # print(a1)
#             # print("PRED")
#             # print(a1_pred)
#             # if len(pred[k]['entities'])==0:
#             #     print(gold_examples[k]['entities'])
#             # print(gold_examples[k]['text'])
#             # print("PRED")
#             # print(pred[k]['entities'])
# print(error_type)
# print(total_entity_type)
# for k in error_type:
#     print(k)
#     print(error_type[k]/total_entity_type[k])