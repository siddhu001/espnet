import argparse
import os
import re
import whisper
from tqdm import tqdm
from nemo_text_processing.text_normalization.normalize import Normalizer


def normalize_text(text):
    text = text.strip()
    text = text.upper()
    text = re.sub(r"[,.!?]", "", text)
    return text


def normalize_text_nemo(text, normalizer):
    text = text.strip()
    text = normalizer.normalize(text)
    text = text.upper()
    text = re.sub(r"[,.!?]", "", text)
    text = text.replace("'S", " 'S")
    return text


def main(args):
    model = whisper.load_model(args.whisper)

    tot_params = sum(p.numel() for p in model.parameters())
    print(f"Total Number of model parameters: {tot_params}")

    with open(os.path.join(args.data_dir, "wav.scp")) as f:
        wavlist = [(l.split()[0], l.split()[1]) for l in f]

    normalizer = Normalizer(input_case="cased")

    results = []
    for index, wav in tqdm(wavlist):
        result = model.transcribe(wav, language="en")
        # norm_text = normalize_text(result["text"])
        norm_text = normalize_text_nemo(result["text"], normalizer=normalizer)
        result = f"{norm_text}\t({index}_1-{index})\n"
        results.append(result)
        print(result, end="")

    os.makedirs(args.exp, exist_ok=True)
    # make `hyp.trn`
    # SET ALARM EVERY MINUTE FOR NEXT HOUR	(alarm_eval_00000001.wav_1-alarm_eval_00000001.wav)
    with open(os.path.join(args.exp, "hyp.trn"), "w") as f:
        f.writelines(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--exp", type=str, default="exp/whisper_large")
    parser.add_argument("--whisper", type=str, default="large")
    args = parser.parse_args()
    main(args)
