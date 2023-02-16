""" Filter augmentated data using decoding results
"""

import argparse
import json
import random
import editdistance

random.seed(0)


def verify_data(hyp, ref, threshold=0.2):
    distance = editdistance.eval(hyp.split(), ref.split())
    err_rate = distance / len(ref.split())
    return err_rate <= threshold


def main(args):
    with open(args.data_path) as f:
        lines = [line.strip() for line in f]
    aug_samples = []
    for line in lines:
        jsonl = json.loads(line)
        aug_samples.append(jsonl)

    with open(args.result_path) as f:
        results = [line.strip() for line in f]

    assert len(results) == len(aug_samples)

    cnt_org, cnt_aug = 0, 0
    jsonls = []
    for result, aug_sample in zip(results, aug_samples):
        if "# " in aug_sample["input"] or "' " in aug_sample["input"]:
            continue

        if "-aug" in aug_sample["index"]:
            if verify_data(result, aug_sample["output"], threshold=args.th):
                jsonls.append(aug_sample)
                cnt_aug += 1
        else:
            # Always add original data
            jsonls.append(aug_sample)
            cnt_org += 1

    print(f"{len(aug_samples):d} -> {len(jsonls):d}")
    print(f"org: {cnt_org:d}, aug: {cnt_aug:d}")

    with open(args.data_path.replace(".json", f"_th{args.th:.2f}.json"), "w") as f:
        for jsonl in jsonls:
            json.dump(jsonl, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("result_path", type=str)
    parser.add_argument("--th", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
