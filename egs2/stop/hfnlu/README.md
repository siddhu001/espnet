# Huggingface transformers NLU

1. (Optional) Set up conda enviroment for huggingface `transformers` (`setup_conda.sh`).

2. Copy `text` (output) and `transcript` (input) from `../asr1/data/{train/valid/test}` to `data/{train/valid/test}`

3. Generate json file
```
python local/preprocess.py
```
The json file will be like:
```
{"index": "alarm_train_00000001.wav", "input": "set the alarm for tonight", "output": "[in:create_alarm [sl:date_time for tonight ] ]"}
{"index": "alarm_train_00000002.wav", "input": "how much time is left before my alarm goes off", "output": "[in:get_alarm ]"}
...
```

4. Train BART-base
```
python run.py conf/bart_base.yaml
```

`run.py` is derived from `run_summarization.py` [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization).

5. (Optional) Predict only
```
python run.py conf/pred_bart_base.yaml
```

6. Calculate EM score
```
python local/score.py exp/bart_base/generated_predictions.txt
```

## Results:

|  | EM on ASR* | EM on GT |
|:---:|:---:|:---:|
| bart_base | --- |  |
| bart_base_v2 | --- |  |
| bart_large | --- |  |
| bart_large_v2 | --- |  |
| bart_base (old) | --- | 81.5 |
| bart_large (old) | 74.7 | 85.0 |

*WER: 3.4
