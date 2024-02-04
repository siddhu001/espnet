import soundfile as sf
import librosa
import kaldiio
from espnet2.bin.s2t_inference_ctc import Speech2Text


s2t = Speech2Text(
    s2t_model_file="exp/s2t_train_s2t_finetune_multitask_ebf_e24_lr2e-4_raw_bpe5000/valid.total_count.ave_10best.pth",
    device="cuda",
    beam_size=1,
    lang_sym="<nolang>",
    task_sym="<asr>"
)


rate, speech = kaldiio.load_mat("/scratch/bbjs/yan2/espnet-ml/egs2/must_c_v2/st1/dump/raw/tst-COMMON.en-de/data/format.1/data_wav.ark:26")

speech, rate = sf.read(
    # "/scratch/bbjs/peng6/SpeechLLM/espnet-slm-new/egs2/speechllm_v1/slm1/ttsmaker-file-2023-11-20-19-40-18.mp3"
    "/scratch/bbjs/peng6/espnet-owsm-ctc/egs2/owsm_ctc_v1/s2t1/covid.wav"
)

speech = librosa.util.fix_length(speech, size=(16000 * 30))

res = s2t(speech, "<na>")[0][0]
