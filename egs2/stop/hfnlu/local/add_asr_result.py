""" Add ASR result from `text` (utt_id+transcript)
"""

import argparse
import json


def main(args):
    with open(args.asr_text) as f:
        input_lines = [line.strip() for line in f]

    ref_path = f"data/{args.dset}/text"
    with open(ref_path) as f:
        output_lines = [line.strip() for line in f]

    jsonls = []

    for inputl, outputl in zip(input_lines, output_lines):
        assert inputl.split()[0] == outputl.split()[0]
        index = inputl.split()[0]

        # NOTE: Use lowercase
        input = " ".join(inputl.split()[1:])
        output = " ".join(outputl.split()[1:])
        input = input.lower()

        if len(input) == 0:
            print(f"{index}: empty hypotheses")
            # NOTE: The model cannot handle empty hypotheses so add `a` instead
            input = "a"

        output = output.lower()

        jsonls.append({"index": index, "input": input, "output": output})

    with open(
        f"data/{args.dset}/data_asr{args.tag}.json", mode="w", encoding="utf-8",
    ) as f:
        for jsonl in jsonls:
            json.dump(
                jsonl, f, ensure_ascii=False,
            )
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("asr_text", type=str)
    parser.add_argument("--dset", type=str, default="test")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()
    main(args)
