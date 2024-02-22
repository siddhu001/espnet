import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report


if __name__ == "__main__":
    pred = "exp/s2t_train_s2t_multitask-ctc_ebf27_conv2d8_size1024_raw_bpe50000/LID_valid.total_count.ave_5best.till40epoch.pth/test/FLEURS/test/text"
    outfile = Path(pred).parent / "result.txt"

    all_true, all_pred = [], []
    with open(pred, "r") as fin:
        for line in fin:
            uttid, full_text = line.strip().split(maxsplit=1)
            lang_true = uttid.split("_")[-2]
            if full_text.startswith("<") and ">" in full_text:
                lang_pred = full_text.split(">")[0].removeprefix("<")
            else:
                lang_pred = "nolang"    # wrong format

            if lang_true == "cmn":
                lang_true = "zho"

            all_true.append(lang_true)
            all_pred.append(lang_pred)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    acc = (all_true == all_pred).astype(np.float32).mean()

    with open(outfile, "w") as fout:
        fout.write(f"Accuracy: {acc}\n")
        fout.write(classification_report(y_true=all_true, y_pred=all_pred))
