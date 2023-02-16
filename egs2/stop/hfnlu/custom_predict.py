import torch
from transformers.utils import logging
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_pt_utils import nested_concat


def predict_with_score(trainer, test_dataset, max_length=128, num_beams=20):
    assert torch.cuda.device_count() == 1

    dataloader = trainer.get_test_dataloader(test_dataset)

    model = trainer.model

    model = trainer._wrap_model(model, training=False, dataloader=dataloader)

    model.eval()

    all_preds, all_scores = None, None

    for step, inputs in enumerate(dataloader):
        inputs = trainer._prepare_inputs(inputs)

        gen_kwargs = {}
        gen_kwargs["max_length"] = max_length
        gen_kwargs["num_beams"] = num_beams
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"]
            if gen_kwargs.get("synced_gpus") is not None
            else default_synced_gpus
        )
        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get(
                "global_attention_mask", None
            )

        gen_inputs = inputs[model.main_input_name]

        gen_output = model.generate(
            gen_inputs,
            **gen_kwargs,
            return_dict_in_generate=True,
            output_scores=True,
            # num_return_sequences=num_beams
        )

        logits = gen_output.sequences  # batch_size*num_return_sequences
        scores = gen_output.sequences_scores

        seq_lens = torch.tensor(
            [sum(logit != 1) for logit in logits], device=logits.device
        )
        scores *= seq_lens

        all_scores = scores if all_scores is None else torch.cat([all_scores, scores])

        logits = trainer._pad_across_processes(logits)
        logits = trainer._nested_gather(logits)
        all_preds = (
            logits
            if all_preds is None
            else nested_concat(all_preds, logits, padding_index=1)
        )

    return all_preds, all_scores
