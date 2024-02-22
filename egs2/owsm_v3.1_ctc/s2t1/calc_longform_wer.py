import os
from pathlib import Path
from whisper.normalizers import EnglishTextNormalizer


def convert_transcript(in_name, out_name, preprocess_fn=None):
    normalizer = EnglishTextNormalizer()
    with open(in_name, "r") as fin, open(out_name, "w") as fout:
        for line in fin:
            uttid, trans = line.strip().split(maxsplit=1)
            if preprocess_fn is not None:
                trans = preprocess_fn(trans)
            trans = normalizer(trans)

            fout.write(f"{trans}\t({uttid})\n")


def preprocess_hyp(line: str):
    invalid = [
        "(Laughter)",
        "(Laughter",
        "Laughter)",
        "(Applause)",
        "(Applause",
        "Applause)",
        "(",
        ")",
    ]
    for pattern in invalid:
        line = line.replace(pattern, "")
    line = line.strip()
    line = " ".join(line.split())
    return line


if __name__ == "__main__":
    hyp = "exp/s2t_train_s2t_multitask-ctc_ebf27_conv2d8_size1024_raw_bpe50000/ASR_TEDLIUM-long_context8_batch32_valid.total_count.ave_5best.till40epoch.pth/text"
    ref = "dump/raw/test/TEDLIUM2_longform/test/text"

    outdir = Path(hyp).parent / "score_wer"
    outdir.mkdir(parents=True, exist_ok=True)

    convert_transcript(
        ref,
        outdir / "ref.trn",
    )

    convert_transcript(
        hyp,
        outdir / "hyp.trn",
        preprocess_fn=preprocess_hyp,
    )

    os.system(
        f"sclite -r {outdir / 'ref.trn'} trn -h {outdir / 'hyp.trn'} trn -i rm -o all stdout > {outdir / 'result.txt'}"
    )
