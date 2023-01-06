import argparse
import json


def get_sp_vocab(data_path):
    sp_vocab_list = []

    with open(data_path) as f:
        for line in f:
            obj = json.loads(line.strip())
            for token in obj["output"].split():
                if token.startswith("[in:") or token.startswith("[sl:"):
                    sp_vocab_list.append(token)

    sp_vocab_list = list(set(sp_vocab_list))
    sp_vocab_list = sorted(sp_vocab_list)

    return sp_vocab_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", type=str)
    parser.add_argument("--valid", type=str)
    parser.add_argument("--test", type=str)
    args = parser.parse_args()
    train_sp_vocab = get_sp_vocab(args.train_path)
    print(train_sp_vocab)

    print("valid")
    valid_sp_vocab = get_sp_vocab(args.valid)
    for v in valid_sp_vocab:
        if v not in train_sp_vocab:
            print(v)

    print("test")
    test_sp_vocab = get_sp_vocab(args.test)
    for v in test_sp_vocab:
        if v not in train_sp_vocab:
            print(v)
