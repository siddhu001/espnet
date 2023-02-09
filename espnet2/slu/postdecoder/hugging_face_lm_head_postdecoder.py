#!/usr/bin/env python3
#  2022, Carnegie Mellon University;  Siddhant Arora
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers PostDecoder."""

from espnet2.slu.postdecoder.abs_postdecoder import AbsPostDecoder

try:
    from transformers import AutoModelWithLMHead, AutoTokenizer

    is_transformers_available = True
except ImportError:
    is_transformers_available = False
import logging

import torch
from typeguard import check_argument_types


class HuggingFaceLMHeadPostDecoder(AbsPostDecoder):
    """Hugging Face Transformers PostEncoder."""

    def __init__(
        self,
        model_name_or_path: str,
        output_size=256,
        dropout_rate=0,
        max_seq_len=None,
        only_encoder=False,
        decoder_input=False,
        decoder_loss=False,
        padding_idx=0,
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

        self.model = AutoModelWithLMHead.from_pretrained(model_name_or_path)
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
        self.decoder_loss=decoder_loss

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
        transcript_outputs = self.model(
            input_ids=transcript_input_ids,
            attention_mask=transcript_attention_mask,
            labels=decoder_input_ids,
            output_hidden_states=True,
        )
        # import pdb;pdb.set_trace()
        final_output=transcript_outputs.decoder_hidden_states[-1]
        if self.decoder_loss:
            return transcript_outputs.loss,self.out_linear(final_output)
        else:
            return self.out_linear(final_output)
    
    def inference(
        self,
        transcript_input_ids: torch.LongTensor,
        transcript_attention_mask: torch.LongTensor,
        decoder_input_ids: torch.LongTensor=None,
        text_labels=None,
    ) -> torch.Tensor:
        """Forward."""
        # import pdb;pdb.set_trace()
        if decoder_input_ids is not None:
            gen_labels=decoder_input_ids
        else:
            transcript_outputs = self.model.generate(transcript_input_ids,
            attention_mask=transcript_attention_mask,max_length=128, num_beams=20, return_dict_in_generate=True, output_hidden_states=True,output_scores=True)
            text_labels=self.tokenizer.batch_decode(transcript_outputs[0], skip_special_tokens=True)
            gen_labels=self.tokenizer(text_labels, return_tensors="pt").input_ids.to(device=transcript_input_ids.device)
        # import pdb;pdb.set_trace()
        transcript_outputs1 = self.model(
            input_ids=transcript_input_ids,
            attention_mask=transcript_attention_mask,
            labels=gen_labels,
            output_hidden_states=True,
        )
        decoder_hidden_best=transcript_outputs1.decoder_hidden_states[-1]
        return text_labels, self.out_linear(decoder_hidden_best)

    def output_size(self) -> int:
        """Get the output size."""
        return self.output_size_dim

    def convert_examples_to_features(self, data, max_seq_length, label_decoding=False, do_padding =True):
        input_id_features = []
        input_mask_features = []
        segment_ids_feature = []
        position_ids_feature = []
        input_id_length = []
        if self.max_seq_len is not None:
            max_seq_length=self.max_seq_len
        for text_id in range(len(data)):
            input_ids = self.tokenizer(data[text_id].lower()).input_ids
            input_mask = [1] * len(input_ids)
            input_id_length.append(len(input_ids))
            # Zero-pad up to the sequence length.
            if do_padding:
                padding = [0] * (max_seq_length - len(input_ids))
                if label_decoding:
                    input_ids += [-100] * (max_seq_length - len(input_ids))
                else:
                    input_ids += padding
                input_mask += padding
            if do_padding:
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
            input_id_features.append(input_ids)
            input_mask_features.append(input_mask)
        return (
            input_id_features,
            input_mask_features,
            segment_ids_feature,
            position_ids_feature,
            input_id_length,
        )
