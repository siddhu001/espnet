#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import argparse
import os
import re
import sys
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

import pandas as pd

def get_classification_result(hyp_file, ref_file, hyp_write, ref_write):
    hyp_lines = [line for line in hyp_file]
    ref_lines = [line for line in ref_file]
    error = 0
    error1=0
    error2=0
    error3 = 0
    hyp_dialog_acts_list = []
    ref_dialog_acts_list = []
    mlb = MultiLabelBinarizer()
    for line_count in range(len(hyp_lines)):
        if "<intent>" not in hyp_lines[line_count]:
            hyp_dialog_acts = []
        elif " <intent>\t" in hyp_lines[line_count]:
            hyp_dialog_acts = hyp_lines[line_count].split(" <intent>\t")[0].split(" <sep> ")
        else:
            hyp_dialog_acts = hyp_lines[line_count].split(" <intent> ")[0].split(" <sep> ")
        ref_dialog_acts = ref_lines[line_count].split(" <intent> ")[0].split(" <sep> ")
        hyp_dialog_acts_list.append(hyp_dialog_acts)
        ref_dialog_acts_list.append(ref_dialog_acts)
        # print(hyp_lines[line_count])
        try:
            hyp_intent = hyp_lines[line_count].split(" <intent> ")[1].split(" <spk_role> ")[0]
        except:
            print(hyp_lines[line_count])
            hyp_intent = ""
        ref_intent = ref_lines[line_count].split(" <intent> ")[1].split(" <spk_role> ")[0]
        try:
            hyp_spk_role = hyp_lines[line_count].split(" <spk_role> ")[1].split(" <spk_id> ")[0]
        except:
            hyp_spk_role = ""
        ref_spk_role = ref_lines[line_count].split(" <spk_role> ")[1].split(" <spk_id> ")[0]
        try:
            hyp_spk_id = hyp_lines[line_count].split(" <spk_id> ")[1].split(" <emotion> ")[0]
        except:
            hyp_spk_id = ""
        ref_spk_id = ref_lines[line_count].split(" <spk_id> ")[1].split(" <emotion> ")[0]
        try:
            hyp_emotion = hyp_lines[line_count].split(" <emotion> ")[1].split(" <utt> ")[0]
        except:
            hyp_emotion = ""
        ref_emotion = ref_lines[line_count].split(" <emotion> ")[1].split(" <utt> ")[0]
        if hyp_emotion != ref_emotion:
            error3 += 1
        if hyp_spk_id != ref_spk_id:
            error2 += 1
        if hyp_spk_role != ref_spk_role:
            error1 += 1
        if hyp_intent != ref_intent:
            error += 1
        try:
            if " <utt> " in hyp_lines[line_count]:
                hyp_write.write(" ".join(hyp_lines[line_count].split(" <intent> ")[1].split(" <utt> ")[1:]))
            else:
                hyp_write.write(" ".join(hyp_lines[line_count].split(" <intent> ")[1].split(" <utt>\t")[1:]))
        except:
            # print(hyp_lines[line_count])
            hyp_write.write("<na> \t"+hyp_lines[line_count].split("\t")[-1])
        ref_write.write(" ".join(ref_lines[line_count].split(" <intent> ")[1].split(" <utt> ")[1:]))
    mlb.fit(ref_dialog_acts_list)
    hyp_dialog_acts_binary = mlb.transform(hyp_dialog_acts_list)
    ref_dialog_acts_binary = mlb.transform(ref_dialog_acts_list)
    print(
        "micro: ",
        f1_score(hyp_dialog_acts_binary, ref_dialog_acts_binary, average="micro"),
    )
    print(
        "macro: ",
        f1_score(hyp_dialog_acts_binary, ref_dialog_acts_binary, average="macro"),
    )
    print("length classes", len(mlb.classes_))
    print(1-(error1 / len(hyp_lines)))
    print(1-(error2 / len(hyp_lines)))
    print(1-(error3 / len(hyp_lines)))
    return 1 - (error / len(hyp_lines))
# def get_classification_result(hyp_file, ref_file, hyp_write, ref_write):
#     hyp_lines = [line for line in hyp_file]
#     ref_lines = [line for line in ref_file]

#     error = 0
#     for line_count in range(len(hyp_lines)):
#         hyp_intent = hyp_lines[line_count].split(" ")[0]
#         ref_intent = ref_lines[line_count].split(" ")[0]
#         if hyp_intent != ref_intent:
#             error += 1
#         hyp_write.write(
#             " ".join(hyp_lines[line_count].split("\t")[0].split(" ")[1:])
#             + "\t"
#             + hyp_lines[line_count].split("\t")[1]
#         )
#         ref_write.write(
#             " ".join(ref_lines[line_count].split("\t")[0].split(" ")[1:])
#             + "\t"
#             + ref_lines[line_count].split("\t")[1]
#         )
#     return 1 - (error / len(hyp_lines))


parser = argparse.ArgumentParser()
parser.add_argument("--exp_root", required=True, help="Directory to save experiments")
parser.add_argument(
    "--valid_folder",
    default="decode_asr_asr_model_valid.acc.ave_10best/devel/",
    help="Directory inside exp_root containing inference on valid set",
)
parser.add_argument(
    "--test_folder",
    default="decode_asr_asr_model_valid.acc.ave_10best/test/",
    help="Directory inside exp_root containing inference on test set",
)
parser.add_argument(
    "--utterance_test_folder",
    default=None,
    help="Directory inside exp_root containing inference on utterance test set",
)

args = parser.parse_args()

exp_root = args.exp_root
valid_inference_folder = args.valid_folder
test_inference_folder = args.test_folder

valid_hyp_file = open(
    os.path.join(exp_root, valid_inference_folder + "score_wer/hyp.trn")
)
valid_ref_file = open(
    os.path.join(exp_root, valid_inference_folder + "score_wer/ref.trn")
)

valid_hyp_write_file = open(
    os.path.join(exp_root, valid_inference_folder + "score_wer/hyp_asr.trn"), "w"
)
valid_ref_write_file = open(
    os.path.join(exp_root, valid_inference_folder + "score_wer/ref_asr.trn"), "w"
)

result = get_classification_result(
    valid_hyp_file, valid_ref_file, valid_hyp_write_file, valid_ref_write_file
)
print("Valid Intent Classification Result")
print(result)

test_hyp_file = open(
    os.path.join(exp_root, test_inference_folder + "score_wer/hyp.trn")
)
test_ref_file = open(
    os.path.join(exp_root, test_inference_folder + "score_wer/ref.trn")
)
test_hyp_write_file = open(
    os.path.join(exp_root, test_inference_folder + "score_wer/hyp_asr.trn"), "w"
)
test_ref_write_file = open(
    os.path.join(exp_root, test_inference_folder + "score_wer/ref_asr.trn"), "w"
)

result = get_classification_result(
    test_hyp_file, test_ref_file, test_hyp_write_file, test_ref_write_file
)
print("Test Intent Classification Result")
print(result)

if args.utterance_test_folder is not None:
    utt_test_inference_folder = args.utterance_test_folder
    utt_test_hyp_file = open(
        os.path.join(exp_root, utt_test_inference_folder + "score_wer/hyp.trn")
    )
    utt_test_ref_file = open(
        os.path.join(exp_root, utt_test_inference_folder + "score_wer/ref.trn")
    )
    utt_test_hyp_write_file = open(
        os.path.join(exp_root, utt_test_inference_folder + "score_wer/hyp_asr.trn"), "w"
    )
    utt_test_ref_write_file = open(
        os.path.join(exp_root, utt_test_inference_folder + "score_wer/ref_asr.trn"), "w"
    )
    result = get_classification_result(
        utt_test_hyp_file,
        utt_test_ref_file,
        utt_test_hyp_write_file,
        utt_test_ref_write_file,
    )
    print("Unseen Utterance Test Intent Classification Result")
    print(result)
