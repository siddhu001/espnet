#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import re
import sys

import pandas as pd

if len(sys.argv) != 2:
    print("Usage: python split_low_resource.py [stoplr_root]")
    sys.exit(1)
stoplr_root = sys.argv[1]

dir_dict = {
    "train-heldin": "held_in_train.tsv",
    "valid-heldin": "held_in_eval.tsv",
    "test-heldin": "held_in_test.tsv",
    "train-reminder-25spis": "reminder_train_25spis.tsv",
    "valid-reminder-25spis": "reminder_valid_25spis.tsv",
    "test-reminder": "held_out_reminder_test.tsv",
    "train-weather-25spis": "weather_train_25spis.tsv",
    "valid-weather-25spis": "weather_valid_25spis.tsv",
    "test-weather": "held_out_weather_test.tsv",
}

for x in dir_dict:
    print("*****", x)
    dataset = f"{x.split('-')[0]}-full"

    text_data = {}
    wav_scp_data = {}
    transcript_data = {}
    utt2spk_data = {}

    with open(os.path.join("data", dataset, "text")) as f:
        text_data = {line.split()[0]: line for line in f}
    with open(os.path.join("data", dataset, "wav.scp")) as f:
        wav_scp_data = {line.split()[0]: line for line in f}
    with open(os.path.join("data", dataset, "transcript")) as f:
        transcript_data = {line.split()[0]: line for line in f}
    with open(os.path.join("data", dataset, "utt2spk")) as f:
        utt2spk_data = {line.split()[0]: line for line in f}

    with open(os.path.join("data", x, "text"), "w") as text_f, open(
        os.path.join("data", x, "wav.scp"), "w"
    ) as wav_scp_f, open(
        os.path.join("data", x, "transcript"), "w"
    ) as transcript_f, open(
        os.path.join("data", x, "utt2spk"), "w"
    ) as utt2spk_f:

        text_f.truncate()
        wav_scp_f.truncate()
        utt2spk_f.truncate()

        # /private/home/padentomasello/data/stop/stop_resampled/
        # train/event_train/00000232.wav	92160
        transcript_df = pd.read_csv(
            os.path.join(stoplr_root, dir_dict[x]), sep="\t", skiprows=1, header=None
        )

        # lines = sorted(transcript_df.values, key=lambda s: s[0])
        for row in transcript_df.values:
            path_arr = row[0].split("/")

            # NOTE: path is fixed in low resource split
            if "eval_0" in path_arr[-3]:
                prefix = path_arr[-2].replace("eval", "eval_0")
            elif "test_0" in path_arr[-3]:
                prefix = path_arr[-2].replace("test", "test_0")
            else:
                prefix = path_arr[-2]

            utt_id = prefix + "_" + path_arr[-1]

            if utt_id not in text_data:
                print(f"{utt_id} not found")
                continue

            text_f.write(text_data[utt_id])
            transcript_f.write(transcript_data[utt_id])
            wav_scp_f.write(wav_scp_data[utt_id])
            utt2spk_f.write(utt2spk_data[utt_id])
