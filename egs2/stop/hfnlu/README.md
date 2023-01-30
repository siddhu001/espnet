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
python run.py conf/bart_large_v7_norms.yaml
```

`run.py` is derived from `run_summarization.py` [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization).

5. Predict
```
python run.py conf/pred_bart_large_v7_norms.yaml
```

6. Scoring
```
bash score.sh exp/bart_large_v7_norms/output.txt
```

## Results:

|  | EM on ASR* | EM on GT |
|:---:|:---:|:---:|
| bart_large_v7 | 79.0 | 85.8 |
| bart_large_v7_norms | 79.8 | 87.1 |

*Whisper ASR (WER=2.4)

## For Track3

```
python run.py conf/bart_large_tr3.yaml
```


