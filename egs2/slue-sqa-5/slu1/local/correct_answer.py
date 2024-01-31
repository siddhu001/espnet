import json
def load_json(fname):
    data = json.loads(open(fname).read())
    return data
from jiwer import wer, cer

pred_dict=load_json("exp/slu_train_asr_wavlm_raw_en_bpe1000/decode_asr_full_path_slu_model_valid.acc.ave/test/test_pred_answer.json")
gold_dict=load_json("data/test/timestamp")
correct=0
total_count=0
WER=0
for uttid in gold_dict:
    gold_arr=[]
    for k in gold_dict[uttid]:
        gold_arr.append(k[0])
    if len(gold_arr)>1:
        print("surprise")
        import pdb;pdb.set_trace()
    total_count+=len(gold_arr)
    if uttid not in pred_dict:
        WER+=1        
    else:
        for k in pred_dict[uttid]:
            # print(uttid)
            # print(k[0])
            # print(gold_arr[0])
            if k[0]=="":
                WER+=1
            else:
                WER+=wer(k[0],gold_arr[0])
                print(wer(k[0],gold_arr[0]))
print(WER/len(gold_dict))

