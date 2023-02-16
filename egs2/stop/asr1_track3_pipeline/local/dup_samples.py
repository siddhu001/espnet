""" Duplicate low-resource samples (upsampling)
"""

import argparse
import os
import shutil


def main(args):
    out_data_dir = args.data_dir.replace("_sp", f"_dup{args.type}{args.times:d}_sp")
    print(f"out_data_dir: {out_data_dir}")

    os.makedirs(out_data_dir, exist_ok=True)
    shutil.copy(
        os.path.join(args.data_dir, "feats_type"),
        os.path.join(out_data_dir, "feats_type"),
    )

    if args.type == "r":
        type_prefix = "reminder_train_"
    elif args.type == "w":
        type_prefix = "weather_train_"

    with open(os.path.join(args.data_dir, "text")) as f:
        lines = [line.strip() for line in f]
    outs = []
    for line in lines:
        wav_org = line.split()[0]
        content = " ".join(line.split()[1:])
        outs.append(f"{wav_org} {content}\n")
        if type_prefix in wav_org:
            for i in range(args.times):
                wav_dup = wav_org.replace(".wav", f"_{(i+1):d}.wav")
                outs.append(f"{wav_dup} {content}\n")
    with open(os.path.join(out_data_dir, "text"), "w") as f:
        f.writelines(outs)

    outs = []
    with open(os.path.join(args.data_dir, "wav.scp")) as f:
        lines = [line.strip() for line in f]
    for line in lines:
        wav_org = line.split()[0]
        src_path = line.split()[1].replace(
            "dump/raw/org/train_sp/",
            f"dump/raw/org/train_dup{args.type}{args.times:d}_sp/",
        )
        outs.append(f"{wav_org} {src_path}\n")
        if type_prefix in wav_org:
            for i in range(args.times):
                wav_dup = wav_org.replace(".wav", f"_{(i+1):d}.wav")
                tgt_path = src_path.replace(".wav.flac", f"_{(i+1):d}.wav.flac")
                shutil.copy(src_path, tgt_path)
                outs.append(f"{wav_dup} {tgt_path}\n")
    with open(os.path.join(out_data_dir, "wav.scp"), "w") as f:
        f.writelines(outs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--times", type=int, default=20)
    parser.add_argument("--type", type=str, choices=["r", "w"], default="r")
    args = parser.parse_args()
    main(args)
