import soundfile as sf
import numpy as np
import librosa
import kaldiio
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch


s2t = Speech2TextGreedySearch(
    s2t_model_file="/scratch/bbjs/peng6/espnet-owsm-ctc-2/egs2/owsm_v3.1_ctc/s2t1/exp/s2t_train_s2t_multitask-ctc_ebf27_conv2d8_size1024_raw_bpe50000/valid.total_count.ave_5best.till40epoch.pth",
    device="cuda",
    generate_interctc_outputs=False,
)


# rate, speech = kaldiio.load_mat(
#     "/scratch/bbjs/yan2/espnet-ml/egs2/must_c_v2/st1/dump/raw/tst-COMMON.en-de/data/format.1/data_wav.ark:26"
# )

# rate, speech = kaldiio.load_mat(
#     "/scratch/bbjs/peng6/espnet-whisper-public/egs2/owsm_v3.1/s2t1/dump/raw/test/AISHELL-1/test/data/format.1/data_wav.ark:696140"
# )

speech, rate = sf.read(
    # "/scratch/bbjs/peng6/espnet-owsm-ctc/egs2/owsm_ctc_v1/s2t1/covid.wav"
    # "shinji.wav"
    "random5s.wav"
    # "ted.wav"
)

# speech = np.zeros((5 * 16000,))

speech = librosa.util.fix_length(speech, size=(16000 * 30))

res = s2t(speech, "<na>", "<eng>", "<asr>")[0]
# res = s2t(speech, " Watanabe ", "<nolang>", "<asr>")[0]
print(res)


# res = s2t.decode_long_batched(speech, batch_size=4, lang_sym="<eng>", task_sym="<asr>")
# print(res)

# res = s2t.decode_long_batched_buffered(
#     speech, batch_size=8, context_len_in_secs=2, lang_sym="<eng>", task_sym="<asr>"
# )
# print(res)
