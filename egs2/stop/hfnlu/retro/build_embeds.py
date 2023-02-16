import argparse
import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TOP_K = 50

ALL_DATASET = "data/train-tr3/"


def main_tfidf(args):
    model = TfidfVectorizer()

    with open(os.path.join(ALL_DATASET, "data.json")) as f:
        lines = [line.strip() for line in f]

    model_path = os.path.join(ALL_DATASET, f"retro_model_{args.tag}.pkl")
    exams_path = os.path.join(ALL_DATASET, f"retro_exams_{args.tag}.pkl")
    embeds_path = os.path.join(ALL_DATASET, f"retro_embeds_{args.tag}.pkl")
    print(f"model_path: {model_path}")
    print(f"exams_path: {exams_path}")
    print(f"embeds_path: {embeds_path}")

    indices, inputs, outputs = [], [], []
    for line in lines:
        jsonl = json.loads(line)
        indices.append(jsonl["index"])
        inputs.append(jsonl["input"])
        outputs.append(jsonl["output"])

    embeds = model.fit_transform(inputs)
    print("embeds:", embeds.shape)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(embeds_path, "wb") as f:
        pickle.dump(embeds, f)

    exams = defaultdict(list)

    size = embeds.shape[0]

    for i, embed in tqdm(enumerate(embeds)):
        cosine_sim = cosine_similarity(embed, embeds)

        topk_ids = np.argsort(cosine_sim[0])[::-1][:TOP_K]

        for j in topk_ids:
            if indices[i] == indices[j]:
                continue

            exams[indices[i]].append((indices[j], inputs[j], outputs[j]))

    with open(exams_path, "wb") as f:
        pickle.dump(exams, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--model", type=str, default="tfidf")
    args = parser.parse_args()
    if args.model == "tfidf":
        main_tfidf(args)
    elif args.model == "st":
        main_st(args)
