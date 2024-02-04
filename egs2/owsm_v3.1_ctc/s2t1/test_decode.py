import soundfile as sf
import librosa
import kaldiio
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch


s2t = Speech2TextGreedySearch(
    s2t_model_file="exp/s2t_train_s2t_multitask-ctc_ebf27_conv2d8_size1024_raw_bpe50000/16epoch.pth",
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
    # "/scratch/bbjs/peng6/SpeechLLM/espnet-slm-new/egs2/speechllm_v1/slm1/ttsmaker-file-2023-11-20-19-40-18.mp3"
    # "/scratch/bbjs/peng6/espnet-owsm-ctc/egs2/owsm_ctc_v1/s2t1/covid.wav"
    "shinji.wav"
)

speech = librosa.util.fix_length(speech, size=(16000 * 30))

# res = s2t(speech, "<na>", "<nolang>", "<asr>")[0]
res = s2t(speech, "Watanabe", "<nolang>", "<asr>")[0]

print(res)
