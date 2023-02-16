import argparse
import json
import pickle
import os
import random
import numpy as np

random.seed(0)
np.random.seed(0)

TOP_K = 50
SEP_TOKEN = "</s>"

ALL_DATASET = "data/train-tr3/"


def main(args):
    probs = np.array([args.p * ((1 - args.p) ** i) for i in range(TOP_K)])

    with open(os.path.join(ALL_DATASET, "data.json")) as f:
        lines = [line.strip() for line in f]

    exams_path = os.path.join(ALL_DATASET, f"retro_exams_{args.tag}.pkl")
    print(f"exams_path: {exams_path}")

    with open(exams_path, "rb") as f:
        exams = pickle.load(f)

    jsonls = []
    for _ in range(args.times):
        for line in lines:
            jsonl = json.loads(line)
            index = jsonl["index"]
            top_k = len(exams[index])
            sample_ids = np.random.choice(
                top_k, args.cat, p=(probs[:top_k] / probs[:top_k].sum()), replace=False
            )
            print(f"* {index} | {jsonl['input']} | {jsonl['output']}")

            input_cat = f"{jsonl['input']} {SEP_TOKEN}"
            for sample_id in sample_ids:
                index_, input_, output_ = exams[index][sample_id]
                print(f">>> {index_} | {input_} | {output_}")
                input_cat += f" {input_} {SEP_TOKEN} {output_} {SEP_TOKEN}"

            jsonls.append(
                {"index": index, "input": input_cat, "output": jsonl["output"]}
            )

    output_path = os.path.join(ALL_DATASET, "data.json").replace(
        ".json", f"_retro_{args.tag}.json"
    )
    output_path = output_path.replace(
        ".json", f"_p{args.p:.2f}_cat{args.cat:d}_x{args.times:d}.json"
    )
    print(output_path)
    with open(output_path, "w",) as f:
        for jsonl in jsonls:
            json.dump(jsonl, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--cat", type=int, default=4)
    parser.add_argument("--times", type=int, default=1)
    args = parser.parse_args()
    main(args)
