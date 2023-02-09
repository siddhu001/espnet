#!/usr/bin/env python3
#  2022, Carnegie Mellon University;  Siddhant Arora
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers PostDecoder."""

from espnet2.slu.postdecoder.abs_postdecoder import AbsPostDecoder

try:
    from transformers import AutoModel, AutoTokenizer

    is_transformers_available = True
except ImportError:
    is_transformers_available = False
import logging

import torch
from typeguard import check_argument_types


class HuggingFaceTransformersPostDecoder(AbsPostDecoder):
    """Hugging Face Transformers PostEncoder."""

    def __init__(
        self,
        model_name_or_path: str,
        output_size=256,
        dropout_rate=0,
        max_seq_len=None,
        only_encoder=False,
        decoder_input=False,
    ):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()
        if not is_transformers_available:
            raise ImportError(
                "`transformers` is not available. Please install it via `pip install"
                " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )

        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
        )
        self.only_encoder=only_encoder
        self.model_name_or_path =  model_name_or_path
        self.max_seq_len=max_seq_len
        self.dropout_rate=dropout_rate
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        logging.info("Pretrained Transformers model parameters reloaded!")
        self.out_linear = torch.nn.Linear(self.model.config.hidden_size, output_size)
        self.output_size_dim = output_size
        self.decoder_input=decoder_input

    def forward(
        self,
        transcript_input_ids: torch.LongTensor,
        transcript_attention_mask: torch.LongTensor,
        transcript_token_type_ids: torch.LongTensor,
        transcript_position_ids: torch.LongTensor,
        decoder_input_ids: torch.LongTensor=None,
        decoder_attention_mask: torch.LongTensor=None,
    ) -> torch.Tensor:
        """Forward."""
        if "bart" in self.model_name_or_path:
            if self.decoder_input:
                transcript_outputs = self.model(
                    input_ids=transcript_input_ids,
                    attention_mask=transcript_attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask
                )
            else:
                transcript_outputs = self.model(
                    input_ids=transcript_input_ids,
                    attention_mask=transcript_attention_mask
                )
        else:
            transcript_outputs = self.model(
                input_ids=transcript_input_ids,
                position_ids=transcript_position_ids,
                attention_mask=transcript_attention_mask,
                token_type_ids=transcript_token_type_ids,
            )
        if self.only_encoder:
            final_output=transcript_outputs.encoder_last_hidden_state
        else:
            final_output=transcript_outputs.last_hidden_state
        if self.dropout_rate>0:
            return self.dropout(self.out_linear(final_output))
        else:
            return self.out_linear(final_output)

    def output_size(self) -> int:
        """Get the output size."""
        return self.output_size_dim

    def convert_examples_to_features(self, data, max_seq_length):
        input_id_features = []
        input_mask_features = []
        segment_ids_feature = []
        position_ids_feature = []
        input_id_length = []
        if self.max_seq_len is not None:
            max_seq_length=self.max_seq_len
        for text_id in range(len(data)):
            tokens_a = self.tokenizer.tokenize(data[text_id])
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[: (max_seq_length - 2)]
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            input_id_length.append(len(input_ids))
            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            position_ids = [i for i in range(max_seq_length)]

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(position_ids) == max_seq_length
            input_id_features.append(input_ids)
            input_mask_features.append(input_mask)
            segment_ids_feature.append(segment_ids)
            position_ids_feature.append(position_ids)
        return (
            input_id_features,
            input_mask_features,
            segment_ids_feature,
            position_ids_feature,
            input_id_length,
        )
