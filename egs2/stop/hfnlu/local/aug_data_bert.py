""" Data augmentation with BERT for low-resource split
"""

import argparse
import json
import random
import re
from tqdm import tqdm

import torch
from transformers import BertTokenizer, BertForMaskedLM

random.seed(0)
torch.random.manual_seed(0)

num2str = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def generate_aug_data(
    jsonl, tokenizer, model, pattern, max_mask_prob=0.3, min_length=5, top_k=10
):
    jsonl = jsonl.copy()

    mask_id = tokenizer._convert_token_to_id("[MASK]")
    apostrophe_id = tokenizer._convert_token_to_id("'")
    mask_prob = random.random() * max_mask_prob

    text_org = jsonl["input"]
    text = text_org + " ."

    tokenized = tokenizer(text)
    input_ids = tokenized["input_ids"]

    indices = [i for i in range(1, len(input_ids) - 2)]

    if not args.v1:
        # NOTE: do not mask apostrophe as it is not masked
        indices = [i for i in indices if input_ids[i] != apostrophe_id]

    if len(indices) < min_length:
        return None

    mask_indices = random.sample(indices, max(int(len(input_ids) * mask_prob), 1))

    input_ids_masked = input_ids.copy()
    input_ids_replaced = input_ids.copy()

    for mask_index in mask_indices:
        input_ids_masked[mask_index] = mask_id
    input_tensor_masked = torch.tensor(input_ids_masked).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor_masked).logits

    for mask_index in mask_indices:
        token = tokenizer._convert_id_to_token(input_ids[mask_index])
        new_token = None

        if args.sample:
            probs = torch.softmax(logits[0, mask_index], dim=-1)
            topk_ids = probs.multinomial(num_samples=top_k)
        else:
            topk_ids = torch.topk(logits[0, mask_index], k=top_k).indices

        for topk_id in topk_ids:
            new_token_ = tokenizer._convert_id_to_token(topk_id.item()).lower()

            if new_token_ in num2str:
                new_token_ = num2str[new_token_]

            if not pattern.match(new_token_):
                continue

            if (not args.v1) and new_token_ == "#":
                continue

            if token.startswith("##"):
                if new_token_.startswith("##"):
                    new_token = new_token_
                    break
            else:
                if not new_token_.startswith("##"):
                    new_token = new_token_
                    break

        if new_token is None:
            new_token = token

        new_token_id = tokenizer._convert_token_to_id(new_token)
        input_ids_replaced[mask_index] = new_token_id

    tokens_replaced = tokenizer.convert_ids_to_tokens(input_ids_replaced)

    tokens_replaced.remove("[CLS]")
    tokens_replaced.remove("[SEP]")
    tokens_replaced.remove(".")

    text_replaced = " ".join(tokens_replaced).replace(" ##", "")
    if not args.v1:
        text_replaced = text_replaced.replace(" ' ", "'")
        text_replaced = text_replaced.replace("'s", " 's")
    text_replaced = text_replaced.lower()

    tokens_org = text_org.split()
    tokens_replaced = text_replaced.split()

    if len(tokens_org) != len(tokens_replaced):
        print(tokens_org, tokens_replaced)
        return None

    output_tokens = jsonl["output"].split()

    for token_org, token_replaced in zip(tokens_org, tokens_replaced):
        if output_tokens.count(token_org) > 2:
            return None
        elif output_tokens.count(token_org) == 1:
            output_tokens[output_tokens.index(token_org)] = token_replaced

    if text_replaced == jsonl["input"]:
        return None

    jsonl["index"] = jsonl["index"] + "-aug"
    jsonl["input"] = text_replaced
    jsonl["output"] = " ".join(output_tokens)

    return jsonl


def main(args):
    if args.v1:
        pattern = re.compile(r"^[a-zA-Z#']+$")
    else:
        # NOTE: `'` is not masked, so can be removed from the pattern
        pattern = re.compile(r"^[a-zA-Z#]+$")

    if args.v1:
        top_k = 5
    else:
        top_k = 10

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    jsonls = []

    with open(args.data_path) as f:
        lines = [line for line in f]

    for line in tqdm(lines):
        for _ in range(args.times):
            jsonl = json.loads(line)
            if args.mask > 0:
                for _ in range(args.aug):
                    jsonl_aug = generate_aug_data(
                        jsonl,
                        tokenizer,
                        model,
                        pattern,
                        max_mask_prob=args.mask,
                        top_k=top_k,
                    )
                    if jsonl_aug is not None:
                        jsonls.append(jsonl_aug)
            jsonls.append(jsonl)

    if args.sample:
        output_path = args.data_path.replace(
            ".json", f"_aug{args.times:d}x{args.aug:d}m{args.mask:.2f}_sample.json"
        )
    else:
        output_path = args.data_path.replace(
            ".json", f"_aug{args.times:d}x{args.aug:d}m{args.mask:.2f}.json"
        )

    if args.v1:
        output_path = output_path.replace(".json", "_v1.json")

    print(f"output: {output_path}")

    with open(output_path, "w",) as f:
        for jsonl in jsonls:
            json.dump(jsonl, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("--mask", type=float, default=0.3)
    parser.add_argument("--times", type=int, default=10)
    parser.add_argument("--aug", type=int, default=1)
    parser.add_argument("--sample", action="store_true")
    # Keep compatiblity to previous version
    parser.add_argument("--v1", action="store_true")
    args = parser.parse_args()
    main(args)
