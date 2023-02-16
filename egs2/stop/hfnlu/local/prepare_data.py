import argparse
import json
import os


def main(args):
    with open(os.path.join(args.data_dir, "transcript")) as f:
        input_lines = [line.strip() for line in f]

    with open(os.path.join(args.data_dir, "text")) as f:
        output_lines = [line.strip() for line in f]

    jsonls = []

    for inputl, outputl in zip(input_lines, output_lines):
        assert inputl.split()[0] == outputl.split()[0]
        index = inputl.split()[0]

        # NOTE: use lowercase
        input = " ".join(inputl.split()[1:])
        output = " ".join(outputl.split()[1:])
        input = input.lower()
        output = output.lower()

        jsonls.append({"index": index, "input": input, "output": output})

    with open(
        os.path.join(args.data_dir, "data.json"), mode="w", encoding="utf-8"
    ) as f:
        for jsonl in jsonls:
            json.dump(
                jsonl, f, ensure_ascii=False,
            )
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()
    main(args)
