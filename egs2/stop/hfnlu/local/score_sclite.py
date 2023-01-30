import argparse
import os


def main(args):
    output_dir = os.path.splitext(args.result_path)[0]
    os.makedirs(output_dir, exist_ok=True)

    ref_path = f"data/{args.dset}/text"

    wavnames = []
    with open(ref_path) as f:
        results = []
        for line in f:
            wavname = line.strip().split()[0]
            wavnames.append(wavname)
            text = " ".join(line.strip().split()[1:])
            result = f"{text}\t({wavname})\n"
            results.append(result)

    with open(os.path.join(output_dir, "ref.trn"), "w") as f:
        f.writelines(results)

    with open(args.result_path) as f:
        results = []
        for index, line in enumerate(f):
            wavname = wavnames[index]
            text = line.strip().upper()  # compare with uppercase
            # 'S -> _'S if not converted
            text = text.replace("'S", " 'S").replace("  ", " ")
            result = f"{text}\t({wavname})\n"
            results.append(result)

    with open(os.path.join(output_dir, "hyp.trn"), "w") as f:
        f.writelines(results)

    assert len(results) == len(wavnames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_path", type=str)
    parser.add_argument("--dset", type=str, default="test")
    args = parser.parse_args()
    main(args)
