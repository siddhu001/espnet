# Huggingface transformers NLU

1. (Optional) Set up conda enviroment for huggingface `transformers` (`setup_conda.sh`).

The following steps are done by `run.sh`

2. Copy `text` (output) and `transcript` (input) from `../asr1/data/{train/valid/test}` to `data/{train/valid/test}`

3. Generate json file
```
python local/prepare_data.py data/train
```
The json file will be like:
```
{"index": "alarm_train_00000001.wav", "input": "set the alarm for tonight", "output": "[in:create_alarm [sl:date_time for tonight ] ]"}
{"index": "alarm_train_00000002.wav", "input": "how much time is left before my alarm goes off", "output": "[in:get_alarm ]"}
...
```

4. Train BART-Large
```
python run.py conf/bart_large_v7.yaml
```

`run.py` is derived from `run_summarization.py` [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization).

5. Predict
```
python run.py conf/pred_bart_large_v7.yaml
```

6. Scoring
```
bash score.sh exp/bart_large_v7/output.txt
```

## Results:

|  | EM on ASR* | EM on GT |
|:---:|:---:|:---:|
| bart_large_v7 (fixed) | 79.9 | 87.1 |

*Whisper ASR (fine-tuned on STOP, WER=2.4)

## For Track3

See `run_track3.sh`

## Results (Track3)

### Reminder

|  | EM on ASR* | EM on GT |
|:---:|:---:|:---:|
| bart_large_tr3r_v2 | - | 67.5 |
| bart_large_tr3r_v2_aug10x1m0.2th0 | - | 69.9 |
| bart_large_tr3r_retro_aug10x1m0.2th0_v2_fix | 63.3 | 73.8 |

*Whisper ASR (fine-tuned on STOP Track3 data, WER=2.8)

### Weather

|  | EM on ASR* | EM on GT |
|:---:|:---:|:---:|
| bart_large_tr3w_v2 | - | 79.7 |
| bart_large_tr3w_v2_aug10x1m0.2th0 | - | 80.3 |
| bart_large_tr3w_retro_aug10x1m0.2th0_v2_fix | 75.0 | 79.7 |

*Whisper ASR (fine-tuned on STOP Track3 data, WER=2.3)
