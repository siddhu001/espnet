import argparse
import os
import random


def get_utt_list(utt2spk_path):
    # collect spk_id and count
    with open(utt2spk_path, "r", encoding="utf-8") as utt2spk:
        spk_dict = {}
        for line in utt2spk.readlines():
            spk_id = line.strip().split()[1]
            if spk_id in spk_dict:
                spk_dict[spk_id] += 1
            else:
                spk_dict[spk_id] = 1
    print(len(spk_dict))
    # orig_split_test_arr=[]
    # for line in open("/ocean/projects/cis210027p/siddhana/harpervalley_splits/val.txt"):
    #     line1=line.strip().replace("_","")
    #     orig_split_test_arr.append(line1)
    # for line in open("/ocean/projects/cis210027p/siddhana/harpervalley_splits/test.txt"):
    #     line1=line.strip().replace("_","")
    #     orig_split_test_arr.append(line1)
    # train_spk_arr=[]
    # for k in spk_dict:
    #     if k not in orig_split_test_arr:
    #         train_spk_arr.append(k)
    # print(len(train_spk_arr))
    conv_val_split_arr=[]
    conv_test_split_arr=[]
    for line in open("/ocean/projects/cis210027p/siddhana/harpervalley_splits/agent_caller_val.db"):
        line1=line.strip().split()[0].split("_")[2]
        if line1 not in conv_val_split_arr:
            conv_val_split_arr.append(line1)
    for line in open("/ocean/projects/cis210027p/siddhana/harpervalley_splits/agent_caller_test.db"):
        line1=line.strip().split()[0].split("_")[2]
        if line1 not in conv_test_split_arr:
            conv_test_split_arr.append(line1)
        if line in conv_val_split_arr:
            print("whaat")
            exit()
    # print(train_spk_arr)
    print(len(conv_val_split_arr))
    print(len(conv_test_split_arr))
    # exit()
    return conv_val_split_arr, conv_test_split_arr


def split_files(source_dir, min_spk_utt, train_frac, val_frac):
    wavscp_train = open("data/train/wav.scp", "w", encoding="utf-8")
    utt2spk_train = open("data/train/utt2spk", "w", encoding="utf-8")
    segments_train = open("data/train/segments", "w", encoding="utf-8")
    text_train = open("data/train/text", "w", encoding="utf-8")
    wavscp_dev = open("data/valid/wav.scp", "w", encoding="utf-8")
    utt2spk_dev = open("data/valid/utt2spk", "w", encoding="utf-8")
    segments_dev = open("data/valid/segments", "w", encoding="utf-8")
    text_dev = open("data/valid/text", "w", encoding="utf-8")
    wavscp_test = open("data/test/wav.scp", "w", encoding="utf-8")
    utt2spk_test = open("data/test/utt2spk", "w", encoding="utf-8")
    segments_test = open("data/test/segments", "w", encoding="utf-8")
    text_test = open("data/test/text", "w", encoding="utf-8")

    dev_conv_list, test_conv_list = get_utt_list(
        os.path.join(source_dir, "utt2spk")
    )

    # split wav.scp
    with open(os.path.join(source_dir, "wav.scp"), "r", encoding="utf-8") as wavscp:
        for line in wavscp.readlines():
            rec_id = line.strip().split()[0]
            spk_id = rec_id.split("-")[0]
            conv_id = rec_id.split("-")[1]
            # print(conv_id)
            if conv_id in dev_conv_list:
                wavscp_dev.write("{}\n".format(line.strip()))
            elif conv_id in test_conv_list:
                wavscp_test.write("{}\n".format(line.strip()))
            else:
                wavscp_train.write("{}\n".format(line.strip()))

    # split utt2spk
    with open(os.path.join(source_dir, "utt2spk"), "r", encoding="utf-8") as utt2spk:
        for line in utt2spk.readlines():
            spk_id = line.strip().split()[1]
            conv_id = line.strip().split()[0].split("-")[1].split("_")[0]
            # print(conv_id)
            if conv_id in dev_conv_list:
                utt2spk_dev.write("{}\n".format(line.strip()))
            elif conv_id in test_conv_list:
                utt2spk_test.write("{}\n".format(line.strip()))
            else:
                utt2spk_train.write("{}\n".format(line.strip()))

    # split segments
    with open(os.path.join(source_dir, "segments"), "r", encoding="utf-8") as segments:
        for line in segments.readlines():
            utt_id = line.strip().split()[0]
            spk_id = utt_id.split("-")[0]
            conv_id =  utt_id.split("-")[1].split("_")[0]
            # print(conv_id)
            if conv_id in dev_conv_list:
                segments_dev.write("{}\n".format(line.strip()))
            elif conv_id in test_conv_list:
                segments_test.write("{}\n".format(line.strip()))
            else:
                segments_train.write("{}\n".format(line.strip()))

    # split text
    with open(os.path.join(source_dir, "text"), "r", encoding="utf-8") as text:
        for line in text.readlines():
            utt_id = line.strip().split()[0]
            spk_id = utt_id.split("-")[0]
            conv_id = utt_id.split("-")[1].split("_")[0]
            if conv_id in dev_conv_list:
                text_dev.write("{}\n".format(line.strip()))
            elif conv_id in test_conv_list:
                text_test.write("{}\n".format(line.strip()))
            else:
                text_train.write("{}\n".format(line.strip()))

    wavscp_train.close()
    utt2spk_train.close()
    segments_train.close()
    text_train.close()
    wavscp_dev.close()
    utt2spk_dev.close()
    segments_dev.close()
    text_dev.close()
    wavscp_test.close()
    utt2spk_test.close()
    segments_test.close()
    text_test.close()


parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, default="data/tmp")
parser.add_argument("--min_spk_utt", type=int, default=10)
parser.add_argument("--train_frac", type=float, default=0.8)
parser.add_argument("--val_frac", type=float, default=0.1)

args = parser.parse_args()

split_files(args.source_dir, args.min_spk_utt, args.train_frac, args.val_frac)
