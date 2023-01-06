import argparse
import json
import os


def main(args):
    with open(os.path.join(args.asr_dir, "text")) as f:
        input_lines = [line.strip() for line in f]

    dset_name = args.asr_dir.rstrip("/").split("/")[-1]
    with open(f"data/{dset_name}/text") as f:
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
        os.path.join(f"data/{dset_name}/data_asr{args.tag}.json"),
        mode="w",
        encoding="utf-8",
    ) as f:
        for jsonl in jsonls:
            json.dump(
                jsonl, f, ensure_ascii=False,
            )
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("asr_dir", type=str)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()
    main(args)
