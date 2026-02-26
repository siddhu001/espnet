#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from pathlib import Path
import random

# Set seed for reproducibility
random.seed(42)
from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset


def extract_target_text_bpe(conversation):
    """Extract entries where text_bpe target becomes true and all preceding entries."""
    # Find the index where text_bpe target becomes true
    target_start_idx = None
    for idx, entry in enumerate(conversation):
        if len(entry) >= 4 and entry[1] == "text_bpe" and entry[2] == True:
            target_start_idx = idx
            break
    
    if target_start_idx is None:
        return None, None, []
    
    # Get all entries before the target entries
    context_entries = []
    for entry in conversation[:target_start_idx]:
        if entry[0]=="user" and entry[1] == "text_bpe":
            continue
        else:
            context_entries.append(entry)
    
    # Get the target entries (where target=True)
    target_entries = []
    for entry in conversation[target_start_idx:]:
        if entry[0]=="user" and entry[1] == "text_bpe":
            continue
        else:
            target_entries.append(entry)
    
    
    return context_entries, target_entries

def process_file(input_file, codec_mapping_file, output_file=None, output_valid_file=None):
    """Process the input JSON file and extract text_bpe entries."""

    # Read input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Group by base key (without _positive/_negative suffix)
    grouped = defaultdict(dict)
    
    base_keys = set()
    for key in data.keys():
        if key.endswith('_positive'):
            base_keys.add(key[:-9])
        elif key.endswith('_negative'):
            base_keys.add(key[:-9])
    
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
        combined = True
        codec_paths = []
        
        # Extract from positive sample
        if 'positive' in samples:
            context_pos,target_pos = extract_target_text_bpe(samples['positive'])
        else:
            combined=False
        
        # Extract from negative sample
        if 'negative' in samples:
            context_neg,target_neg = extract_target_text_bpe(samples['negative'])
        else:
            combined=False
        
        # Verify context is the same
        if context_pos != context_neg:
            print(f"Warning: Context mismatch for {base_key}")
            continue
        
        if combined:
            dialogue = Dialogue(task="audio_dialogue")
            for k in context_pos:
                dialogue.add_segment(
                    role=k[0],
                    modality=k[1],
                    target=False,
                    content=k[-1],
                )
            for k in target_pos:
                if k[1]=="codec_ssl":
                    dialogue.add_segment(
                        role=k[0],
                        modality=k[1],
                        target=False,
                        content=k[-1],
                    )
                else:
                    dialogue.add_segment(
                        role=k[0],
                        modality=k[1],
                        target=k[2],
                        content=k[-1],
                    )
            for k in context_pos:
                dialogue.add_segment(
                    role=k[0],
                    modality=k[1],
                    target=False,
                    content=k[-1],
                )
            for k in target_neg:
                if k[1]=="codec_ssl":
                    dialogue.add_segment(
                        role=k[0],
                        modality=k[1],
                        target=False,
                        content=k[-1],
                    )
                else:
                    dialogue.add_segment(
                        role=k[0],
                        modality=k[1],
                        target=k[2],
                        content=k[-1],
                    )
            example_id = base_key
            train_dataset.add_dialogue(example_id, dialogue)
    
    # Output result
    output_dir=Path(output_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset.dump_dataset(output_dir)
    
    for base_key, samples in grouped_valid.items():
        combined = True
        codec_paths = []
        
        # Extract from positive sample
        if 'positive' in samples:
            context_pos,target_pos = extract_target_text_bpe(samples['positive'])
        else:
            combined=False
        
        # Extract from negative sample
        if 'negative' in samples:
            context_neg,target_neg = extract_target_text_bpe(samples['negative'])
        else:
            combined=False
        
        # Verify context is the same
        if context_pos != context_neg:
            print(f"Warning: Context mismatch for {base_key}")
            continue
        
        if combined:
            dialogue = Dialogue(task="audio_dialogue")
            for k in context_pos:
                dialogue.add_segment(
                    role=k[0],
                    modality=k[1],
                    target=False,
                    content=k[-1],
                )
            for k in target_pos:
                if k[1]=="codec_ssl":
                    dialogue.add_segment(
                        role=k[0],
                        modality=k[1],
                        target=False,
                        content=k[-1],
                    )
                else:
                    dialogue.add_segment(
                        role=k[0],
                        modality=k[1],
                        target=k[2],
                        content=k[-1],
                    )
            for k in context_pos:
                dialogue.add_segment(
                    role=k[0],
                    modality=k[1],
                    target=False,
                    content=k[-1],
                )
            for k in target_neg:
                if k[1]=="codec_ssl":
                    dialogue.add_segment(
                        role=k[0],
                        modality=k[1],
                        target=False,
                        content=k[-1],
                    )
                else:
                    dialogue.add_segment(
                        role=k[0],
                        modality=k[1],
                        target=k[2],
                        content=k[-1],
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