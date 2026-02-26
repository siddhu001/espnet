#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from pathlib import Path
import random

# Set seed for reproducibility
random.seed(42)
from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset

def load_codec_mapping(codec_mapping_file):
    """Load codec_ssl path mapping from file."""
    codec_mapping = {}
    with open(codec_mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                key = parts[0]
                path = parts[1]
                codec_mapping[key] = path
    return codec_mapping

def extract_last_text_bpe(conversation):
    """Extract the last user and assistant text_bpe entries from a conversation."""
    last_user = None
    last_assistant = None
    
    # Iterate through conversation to find last user and assistant text_bpe entries
    total_entry=[]
    for entry in conversation[:-1]:
        if entry[1] == "codec_ssl":
            total_entry.append(entry)
        if len(entry) >= 4 and entry[1] == "text_bpe":
            total_entry.append(entry)
            if entry[0] == "user":
                last_user = entry
            elif entry[0] == "assistant":
                last_assistant = entry
    
    return last_user, last_assistant, total_entry[:-2]

def process_file(input_file, codec_mapping_file, output_file=None, output_valid_file=None):
    """Process the input JSON file and extract text_bpe entries."""

    codec_mapping = load_codec_mapping(codec_mapping_file)
    # Read input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Group by base key (without _positive/_negative suffix)
    grouped = defaultdict(dict)
    
    base_keys = set()
    for key in data.keys():
        if key not in codec_mapping:
            continue
        if key.endswith('_positive'):
            base_keys.add(key[:-9])
        elif key.endswith('_negative'):
            base_keys.add(key[:-9])
    import pdb;pdb.set_trace()

    base_keys_list = sorted(list(base_keys))  # Sort for reproducibility
    random.shuffle(base_keys_list)

    split_idx = len(base_keys_list) - 361
    train_keys = set(base_keys_list[:split_idx])
    valid_keys = set(base_keys_list[split_idx:])

    # Create separate grouped dictionaries for train and valid
    grouped_train = defaultdict(dict)
    grouped_valid = defaultdict(dict)

    # Second pass: assign to train or valid
    for key, conversation in data.items():
        if key.endswith('_positive'):
            base_key = key[:-9]
            if base_key in train_keys:
                grouped_train[base_key]['positive'] = conversation
                grouped_train[base_key]['positive_key'] = key
            elif base_key in valid_keys:
                grouped_valid[base_key]['positive'] = conversation
                grouped_valid[base_key]['positive_key'] = key
        elif key.endswith('_negative'):
            base_key = key[:-9]
            if base_key in train_keys:
                grouped_train[base_key]['negative'] = conversation
                grouped_train[base_key]['negative_key'] = key
            elif base_key in valid_keys:
                grouped_valid[base_key]['negative'] = conversation
                grouped_valid[base_key]['negative_key'] = key
    # Process each group
    
    train_dataset = DialogueDataset(task="audio_dialogue")
    valid_dataset = DialogueDataset(task="audio_dialogue")
    for base_key, samples in grouped_train.items():
        combined = []
        codec_paths = []
        
        
        # Extract from positive sample
        if 'positive' in samples:
            user_pos, assistant_pos, total_pos = extract_last_text_bpe(samples['positive'])
            if user_pos and assistant_pos:
                combined.append(user_pos)
                combined.append(assistant_pos)
                pos_key = samples['positive_key']
                codec_paths.append(codec_mapping[pos_key])
        
        # Extract from negative sample
        if 'negative' in samples:
            user_neg, assistant_neg, total_neg = extract_last_text_bpe(samples['negative'])
            if user_neg and assistant_neg:
                combined.append(user_neg)
                combined.append(assistant_neg)
                neg_key = samples['negative_key']
                codec_paths.append(codec_mapping[neg_key])
        assert total_pos==total_neg
        
        if combined:
            assert len(combined)==4
            dialogue = Dialogue(task="audio_dialogue")
            for k in total_pos:
                dialogue.add_segment(
                    role=k[0],
                    modality=k[1],
                    target=False,
                    content=k[-1],
                )
            dialogue.add_segment(
                role=combined[0][0],
                modality='text_bpe',
                target=False,
                content=combined[0][-1],
            )
            dialogue.add_segment(
                role=combined[1][0],
                modality='text_bpe',
                target=True,
                content=combined[1][-1],
            )
            dialogue.add_segment(
                role="assistant",
                modality='codec_ssl',
                target=False,
                content=codec_paths[0],
            )
            for k in total_pos:
                dialogue.add_segment(
                    role=k[0],
                    modality=k[1],
                    target=False,
                    content=k[-1],
                )
            dialogue.add_segment(
                role=combined[0][0],
                modality='text_bpe',
                target=False,
                content=combined[0][-1],
            )
            dialogue.add_segment(
                role=combined[-1][0],
                modality='text_bpe',
                target=True,
                content=combined[-1][-1],
            )
            dialogue.add_segment(
                role="assistant",
                modality='codec_ssl',
                target=False,
                content=codec_paths[1],
            )
            example_id = base_key
            train_dataset.add_dialogue(example_id, dialogue)
    
    # Output result
    output_dir=Path(output_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset.dump_dataset(output_dir)
    
    for base_key, samples in grouped_valid.items():
        combined = []
        codec_paths = []
        
        # Extract from positive sample
        if 'positive' in samples:
            user_pos, assistant_pos, total_pos = extract_last_text_bpe(samples['positive'])
            if user_pos and assistant_pos:
                combined.append(user_pos)
                combined.append(assistant_pos)
                pos_key = samples['positive_key']
                codec_paths.append(codec_mapping[pos_key])
        
        # Extract from negative sample
        if 'negative' in samples:
            user_neg, assistant_neg, total_neg = extract_last_text_bpe(samples['negative'])
            if user_neg and assistant_neg:
                combined.append(user_neg)
                combined.append(assistant_neg)
                neg_key = samples['negative_key']
                codec_paths.append(codec_mapping[neg_key])
        
        if combined:
            assert len(combined)==4
            dialogue = Dialogue(task="audio_dialogue")
            for k in total_pos:
                dialogue.add_segment(
                    role=k[0],
                    modality=k[1],
                    target=False,
                    content=k[-1],
                )
            dialogue.add_segment(
                role=combined[0][0],
                modality='text_bpe',
                target=False,
                content=combined[0][-1],
            )
            dialogue.add_segment(
                role=combined[1][0],
                modality='text_bpe',
                target=True,
                content=combined[1][-1],
            )
            dialogue.add_segment(
                role="assistant",
                modality='codec_ssl',
                target=False,
                content=codec_paths[0],
            )
            for k in total_pos:
                dialogue.add_segment(
                    role=k[0],
                    modality=k[1],
                    target=False,
                    content=k[-1],
                )
            dialogue.add_segment(
                role=combined[0][0],
                modality='text_bpe',
                target=False,
                content=combined[0][-1],
            )
            dialogue.add_segment(
                role=combined[-1][0],
                modality='text_bpe',
                target=True,
                content=combined[-1][-1],
            )
            dialogue.add_segment(
                role="assistant",
                modality='codec_ssl',
                target=False,
                content=codec_paths[1],
            )
            example_id = base_key
            valid_dataset.add_dialogue(example_id, dialogue)
    
    # Output result
    output_valid_dir=Path(output_valid_file)
    output_valid_dir.mkdir(parents=True, exist_ok=True)
    valid_dataset.dump_dataset(output_valid_dir)
    
    return

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python extract_text_bpe.py <input_file> <codec_mapping_file> <output_train_dir> <output_valid_dir>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    codec_mapping_file = sys.argv[2]
    output_dir = sys.argv[3]
    output_valid_dir = sys.argv[4]
    
    process_file(input_file, codec_mapping_file, output_dir, output_valid_dir)