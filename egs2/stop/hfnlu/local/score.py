import argparse
import os


def main(args):
    with open(args.result_path) as f:
        preds = [line.strip().upper() for line in f]

    with open(args.ref) as f:
        indices, trues = [], []
        for line in f:
            indices.append(line.strip().split()[0])
            trues.append(" ".join(line.strip().split()[1:]))

    with open(os.path.join(os.path.dirname(args.result_path), "result"), "w") as f:
        for index, pred in zip(indices, preds):
            f.write(f"{index} {pred}\n")

    print(len(preds))
    print(len(trues))
    assert len(preds) == len(trues)

    cnt_em = 0
    for index, pred, true in zip(indices, preds, trues):
        print(index + ":" + str(pred == true))
        print(pred)
        print(true)
        if pred == true:
            cnt_em += 1

    print("==========")
    print(len(trues))
    print(cnt_em / len(trues))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_path", type=str)
    parser.add_argument("--ref", type=str, default="data/test/text")
    args = parser.parse_args()
    main(args)
