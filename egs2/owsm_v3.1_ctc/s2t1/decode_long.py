import logging
import soundfile as sf
from pathlib import Path
import torch

from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch


if __name__ == "__main__":
    # beam_size = 1
    # condition_on_prev_text = False
    context_len_in_secs = 8
    batch_size = 32
    s2t_model_file = "exp/s2t_train_s2t_multitask-ctc_ebf27_conv2d8_size1024_raw_bpe50000/valid.total_count.ave_5best.till40epoch.pth"
    s2t = Speech2TextGreedySearch(
        s2t_model_file=s2t_model_file,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        generate_interctc_outputs=False,
        lang_sym='<eng>',
        task_sym='<asr>',
    )

    out_dir = Path(s2t_model_file).parent / f"ASR_TEDLIUM-long_context{context_len_in_secs}_batch{batch_size}_{s2t_model_file.split('/')[-1]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=out_dir / "decode.1.log",
        filemode='a',
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        force=True,
    )

    data_dir = "dump/raw/test/TEDLIUM2_longform/test"
    with open(Path(data_dir) / "wav.scp", 'r') as fp, open(
        out_dir / "text", 'w'
    ) as fout:
        for line in fp:
            uttid, wavpath = line.strip().split(maxsplit=1)
            speech, _ = sf.read(wavpath)

            logging.info(f"uttid: {uttid}")
            logging.info(f"speech length: {len(speech)}")
            text = s2t.decode_long_batched_buffered(
                speech,
                batch_size=batch_size,
                context_len_in_secs=context_len_in_secs,
                frames_per_sec=12.5,        # 80ms shift
            )
            logging.info(f"best hypo: {text}")
            fout.write(f"{uttid} {text}\n")
