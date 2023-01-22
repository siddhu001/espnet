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


## Results:

|  | EM on ASR* | EM on GT |
|:---:|:---:|:---:|
| bart_base | --- | 81.6 |
| bart_base_v2 | --- | 81.1 |
| bart_large** | --- | 84.9 |
| bart_large_v7 | 78.1 | 85.8 |

*WavLM WER: 2.7
**Training curve looks unstable
