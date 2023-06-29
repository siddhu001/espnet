from local.evaluation.util import load_gold_data, load_predictions
gold_examples = load_gold_data("/scratch/bbjs/arora1/slurp/dataset/slurp/test.jsonl",False)
in_arr=["play_music",
"iot_hue_lightoff",
"play_radio",
"weather_query",
"cooking_recipe",
"email_sendemail",
"calendar_set",
"lists_remove",
"play_podcasts",
"qa_definition",
"audio_volume_up",
"news_query",
"general_quirky",
"email_query",
"audio_volume_down",
"takeaway_query",
"general_joke",
"iot_hue_lightup",
"takeaway_order",
"audio_volume_mute",
"iot_hue_lightdim",
"calendar_query",
"transport_query",
"transport_taxi",
"general_greet",
"music_query",
"iot_coffee",
"qa_maths",
"email_querycontact",
"recommendation_movies",
"alarm_remove",
"calendar_remove",
"datetime_query",
"iot_hue_lightchange",
"iot_wemo_off",
"transport_ticket",
"alarm_query",
"transport_traffic",
"recommendation_events",
"lists_createoradd",
"social_query",
"social_post",
"qa_stock",
"lists_query",
"qa_factoid",
"recommendation_locations",
"audio_volume_other",
"qa_currency",
"iot_cleaning",
"play_audiobook",
"alarm_set",
"datetime_convert",
"play_game",
"iot_wemo_on",
"music_dislikeness",
"email_addcontact",
"iot_hue_lighton",
"cooking_query",
"qa_query",
"general_negate",
"general_dontcare",
"general_repeat",
"general_affirm",
"general_commandstop",
"general_confirm",
"general_explain",
"general_praise"]
pred_nostop = load_predictions("../asr2_combined/exp/asr_train_asr_whisper_full_correct_specaug_raw_en_whisper_multilingual/result_test.json",False)
pred = load_predictions("exp/asr_train_asr_whisper_full_correct_specaug_raw_en_whisper_multilingual/result_test.json",False)
pred_transcript_file=open("exp/asr_train_asr_whisper_full_correct_specaug_raw_en_whisper_multilingual/decode_asr_ner_asr_model_valid.acc.ave/test_slurp/text")
pred_transcript_arr=[line.strip() for line in pred_transcript_file]
pred_transcript_dict={}
for line in pred_transcript_arr:
    pred_transcript_dict["audio-"+line.split()[0].split("_")[-1]+".flac"]=line.split(" SEP ")[-1]
total_entity_type={}
error_type={}
for k in gold_examples:
    a1="{}_{}".format(gold_examples[k]["scenario"], gold_examples[k]["action"])
    a1_pred="{}_{}".format(pred[k]["scenario"], pred[k]["action"])
    a1_pred_nostop="{}_{}".format(pred_nostop[k]["scenario"], pred_nostop[k]["action"])
    # if a1==a1_pred_nostop:
    #     if a1!=a1_pred:
    #         # if gold_examples[k]["text"]==pred_transcript_dict[k]:
    #         if a1_pred not in in_arr:
    #             print(gold_examples[k]["text"])
    #             print(a1)
    #             print("PRED")
    #             print(pred_transcript_dict[k])
    #             # print(pred[k]["text"])
    #             print(a1_pred)
    if gold_examples[k]["entities"]==pred_nostop[k]["entities"]:
        if gold_examples[k]["entities"]!=pred[k]["entities"]:
            # print(gold_examples[k]["entities"])
            # print(pred[k]["entities"])
            if len(pred[k]['entities'])==0:
                print(gold_examples[k]['entities'])
            # print(gold_examples[k]['text'])
            print("PRED")
            # print(pred[k]['entities'])
# print(error_type)
# print(total_entity_type)
# for k in error_type:
#     print(k)
#     print(error_type[k]/total_entity_type[k])