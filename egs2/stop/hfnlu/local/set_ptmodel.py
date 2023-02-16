import argparse
import os
import shutil


def main(args):
    ckpts = [
        ckpt for ckpt in os.listdir(args.exp_dir) if ckpt.startswith("checkpoint-")
    ]
    ckpts.sort(key=lambda x: int(x.split("-")[-1]))
    ckpt_src = os.path.join(args.exp_dir, ckpts[-1])
    ckpt_tgt = os.path.join(args.exp_dir, "checkpoint-1")
    os.makedirs(ckpt_tgt)
    print(ckpt_src, "->", ckpt_tgt)

    shutil.copy(os.path.join(ckpt_src, "config.json"), os.path.join(ckpt_tgt, "."))
    shutil.copy(
        os.path.join(ckpt_src, "pytorch_model.bin"), os.path.join(ckpt_tgt, ".")
    )
    shutil.copy(os.path.join(ckpt_src, "tokenizer.json"), os.path.join(ckpt_tgt, "."))
    shutil.copy(os.path.join(ckpt_src, "vocab.json"), os.path.join(ckpt_tgt, "."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=str)
    args = parser.parse_args()
    main(args)
