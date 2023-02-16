import argparse
import json
import pickle
import os
import random
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

random.seed(0)
np.random.seed(0)

TOP_K = 50
SEP_TOKEN = "</s>"

DATABASE = "data/train-tr3/"


def main(args):
    probs = np.array([args.p * ((1 - args.p) ** i) for i in range(TOP_K)])

    with open(os.path.join(DATABASE, "data.json")) as f:
        lines = [line.strip() for line in f]

    indices_db = [json.loads(line)["index"] for line in lines]
    inputs_db = [json.loads(line)["input"] for line in lines]
    outputs_db = [json.loads(line)["output"] for line in lines]

    if args.model == "tfidf":
        model_path = os.path.join(DATABASE, f"retro_model_{args.tag}.pkl")
        embeds_path = os.path.join(DATABASE, f"retro_embeds_{args.tag}.pkl")
        print(f"model_path: {model_path}")
        print(f"embeds_path: {embeds_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    elif args.model == "st":
        embeds_path = os.path.join(DATABASE, f"retro_st_embeds_{args.tag}.pkl")
        print(f"embeds_path: {embeds_path}")
        model = SentenceTransformer("all-MiniLM-L6-v2")

    with open(embeds_path, "rb") as f:
        embeds_db = pickle.load(f)

    with open(args.data_path) as f:
        lines = [line.strip() for line in f]

    jsonls = []
    for _ in tqdm(range(args.times)):
        for line in lines:
            jsonl = json.loads(line)

            if args.model == "tfidf":
                embed = model.transform([jsonl["input"]])
            elif args.model == "st":
                embed = model.encode([jsonl["input"]])

            cosine_sim = cosine_similarity(embed, embeds_db)

            if args.train:
                topk_ids = np.argsort(cosine_sim[0])[::-1][:TOP_K]
                top_k = len(topk_ids)
                sample_ids_ = np.random.choice(
                    top_k,
                    args.cat + 1,
                    p=(probs[:top_k] / probs[:top_k].sum()),
                    replace=False,
                )
                sample_ids = [topk_ids[idx] for idx in sample_ids_]
            else:
                sample_ids = np.argsort(cosine_sim[0])[::-1][: (args.cat + 1)]

            input_cat = f"{jsonl['input']} {SEP_TOKEN}"

            print(f"* {jsonl['index']} | {jsonl['input']} | {jsonl['output']}")

            cnt_cat = 0
            for sample_id in sample_ids:
                if cnt_cat >= args.cat:
                    continue
                if not args.train:
                    assert jsonl["index"] != indices_db[sample_id]

                index = jsonl["index"]
                if args.rmaug:
                    index = index.replace("-aug", "")
                if index == indices_db[sample_id]:
                    continue

                index_ = indices_db[sample_id]
                input_ = inputs_db[sample_id]
                output_ = outputs_db[sample_id]
                print(f">>> {index_} | {input_} | {output_}")
                input_cat += f" {input_} {SEP_TOKEN} {output_} {SEP_TOKEN}"
                cnt_cat += 1

            assert cnt_cat == args.cat

            jsonls.append(
                {
                    "index": jsonl["index"],
                    "input": input_cat,
                    "output": jsonl["output"],
                }
            )

    if args.model == "tfidf":
        output_path = args.data_path.replace(".json", f"_retro_{args.tag}.json")
    elif args.model == "st":
        output_path = args.data_path.replace(".json", f"_retro_st_{args.tag}.json")

    if args.train:
        output_path = output_path.replace(
            ".json", f"_p{args.p:.2f}_cat{args.cat:d}_x{args.times:d}.json"
        )
        if args.rmaug:
            output_path = output_path.replace(".json", f"_rmaug.json")
    else:
        output_path = output_path.replace(".json", f"_cat{args.cat:d}.json")
    print(output_path)
    with open(output_path, "w",) as f:
        for jsonl in jsonls:
            json.dump(jsonl, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--cat", type=int, default=4)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--rmaug", action="store_true")
    parser.add_argument("--model", type=str, default="tfidf")
    args = parser.parse_args()
    if not args.train:
        assert args.times == 1
    main(args)
