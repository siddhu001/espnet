import shutil
from pathlib import Path


def convert_st_dir(
    in_dir,
    out_dir,
    asr_suffix,
    st_suffix,
    src_lang,
    tgt_lang,
    tasks=['asr', 'st'],
):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(
        in_dir / "feats_type",
        out_dir / "feats_type",
    )

    fin_asr_text = (in_dir / f"text{asr_suffix}").open("r")
    fin_st_text = (in_dir / f"text{st_suffix}").open("r")
    fin_spk = (in_dir / "utt2spk").open("r")
    fin_wav = (in_dir / "wav.scp").open("r")

    fout_text = (out_dir / "text").open("w")
    fout_text_ctc = (out_dir / "text.ctc").open("w")
    fout_text_prev = (out_dir / "text.prev").open("w")
    fout_spk = (out_dir / "utt2spk").open("w")
    fout_wav = (out_dir / "wav.scp").open("w")

    for asr_text, st_text, spk, wav in zip(fin_asr_text, fin_st_text, fin_spk, fin_wav):
        uttid, asr_text = asr_text.split(maxsplit=1)
        _, st_text = st_text.split(maxsplit=1)
        _, spk = spk.split(maxsplit=1)
        _, wav = wav.split(maxsplit=1)

        if 'asr' in tasks:
            fout_text.write(f"{uttid}_{src_lang}_asr <{src_lang}><asr> {asr_text}")
            fout_text_ctc.write(f"{uttid}_{src_lang}_asr {asr_text}")
            fout_text_prev.write(f"{uttid}_{src_lang}_asr <na>\n")
            fout_spk.write(f"{uttid}_{src_lang}_asr {spk}")
            fout_wav.write(f"{uttid}_{src_lang}_asr {wav}")
        
        if 'st' in tasks:
            fout_text.write(f"{uttid}_{src_lang}_st_{tgt_lang} <{src_lang}><st_{tgt_lang}> {st_text}")
            fout_text_ctc.write(f"{uttid}_{src_lang}_st_{tgt_lang} {asr_text}")
            fout_text_prev.write(f"{uttid}_{src_lang}_st_{tgt_lang} <na>\n")
            fout_spk.write(f"{uttid}_{src_lang}_st_{tgt_lang} {spk}")
            fout_wav.write(f"{uttid}_{src_lang}_st_{tgt_lang} {wav}")


if __name__ == "__main__":
    convert_st_dir(
        "dump/raw/dev.en-de",
        "dump/raw/owsm_dev.en-de",
        ".lc.rm.en",
        ".tc.de",
        "en",
        "de",
        ['st'],
    )

    convert_st_dir(
        "dump/raw/train.en-de_sp",
        "dump/raw/owsm_train.en-de_sp",
        ".lc.rm.en",
        ".tc.de",
        "en",
        "de",
        ['st'],
    )
