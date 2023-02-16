""" Convert _'s -> 's (e.g. what 's -> what's)
"""

import argparse
import json


def main(args):
    jsonls = []

    with open(args.json_path) as f:
        for line in f:
            jsonl = json.loads(line)
            jsonl["input"] = jsonl["input"].replace(" 's", "'s")
            jsonl["output"] = jsonl["output"].replace(" 's", "'s")
            jsonls.append(jsonl)

    with open(args.json_path.replace(".json", "_norms.json"), "w") as f:
        for jsonl in jsonls:
            json.dump(jsonl, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=str)
    args = parser.parse_args()
    main(args)
