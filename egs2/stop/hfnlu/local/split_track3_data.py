import argparse
import json
import os
import pandas as pd


LR_TSVS = {
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


def main(args):
    utt_ids_all_train = []
    utt_ids_all_valid = []

    for key, tsv in LR_TSVS.items():
        lr_tsv_path = os.path.join(args.lr_splits, tsv)
        print(lr_tsv_path)
        df = pd.read_csv(lr_tsv_path, sep="\t", skiprows=1, header=None)
        print(len(df))

        utt_ids = []
        for row in df.values:
            path = row[0].split("/")

            if "eval_0" in path[-3]:
                prefix = path[-2].replace("eval", "eval_0")
            elif "test_0" in path[-3]:
                prefix = path[-2].replace("test", "test_0")
            else:
                prefix = path[-2]

            utt_id = prefix + "_" + path[-1]

            utt_ids.append(utt_id)

            if "train-" in key:
                utt_ids_all_train.append(utt_id)
            if "valid-" in key:
                utt_ids_all_valid.append(utt_id)

        jsonls = []
        json_path = os.path.join(args.data, key.split("-")[0], "data.json")

        os.makedirs(os.path.join(args.outdata, key), exist_ok=True)
        out_json_path = os.path.join(args.outdata, key, "data.json")
        print(json_path, "->", out_json_path)
        with open(json_path) as f:
            for line in f:
                jsonl = json.loads(line)
                if jsonl["index"] in utt_ids:
                    jsonls.append(jsonl)
        print(len(jsonls))

        with open(out_json_path, "w") as f:
            for jsonl in jsonls:
                json.dump(jsonl, f, ensure_ascii=True)
                f.write("\n")

    # train-tr3
    json_path = os.path.join(args.data, "train", "data.json")
    jsonls = []
    os.makedirs(os.path.join(args.outdata, "train-tr3"), exist_ok=True)
    out_json_path = os.path.join(args.outdata, "train-tr3", "data.json")
    print(json_path, "->", out_json_path)
    with open(json_path) as f:
        for line in f:
            jsonl = json.loads(line)
            if jsonl["index"] in utt_ids_all_train:
                jsonls.append(jsonl)
    print(len(jsonls))
    with open(out_json_path, "w") as f:
        for jsonl in jsonls:
            json.dump(jsonl, f, ensure_ascii=True)
            f.write("\n")

    # valid-tr3
    json_path = os.path.join(args.data, "valid", "data.json")
    jsonls = []
    os.makedirs(os.path.join(args.outdata, "valid-tr3"), exist_ok=True)
    out_json_path = os.path.join(args.outdata, "valid-tr3", "data.json")
    print(json_path, "->", out_json_path)
    with open(json_path) as f:
        for line in f:
            jsonl = json.loads(line)
            if jsonl["index"] in utt_ids_all_valid:
                jsonls.append(jsonl)
    print(len(jsonls))
    with open(out_json_path, "w") as f:
        for jsonl in jsonls:
            json.dump(jsonl, f, ensure_ascii=True)
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lr_splits", type=str)
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--outdata", type=str, default="data")
    args = parser.parse_args()
    main(args)
