#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from pathlib import Path
import random
import re

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

def parse_key(key):
    """
    Parse key to extract base_key, type (positive/negative), and index.
    Examples:
        'key_positive_0' -> ('key', 'positive', 0)
        'key_negative_2' -> ('key', 'negative', 2)
    """
    # Match pattern: <base_key>_positive_<idx> or <base_key>_negative_<idx>
    match = re.match(r'^(.+)_(positive|negative)_(\d+)$', key)
    if match:
        base_key = match.group(1)
        sample_type = match.group(2)
        idx = int(match.group(3))
        return base_key, sample_type, idx
    return None, None, None

def process_file(input_file, codec_mapping_file, output_file=None, output_valid_file=None):
    """Process the input JSON file and extract text_bpe entries."""

    # codec_mapping = load_codec_mapping(codec_mapping_file)
    
    # Read input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Group by base key and index
    # Structure: grouped[base_key][idx] = {'positive': conv, 'negative': conv, 'positive_key': key, 'negative_key': key}
    grouped = defaultdict(lambda: defaultdict(dict))
    
    base_keys = set()
    
    for key, conversation in data.items():
        # if key not in codec_mapping:
            # continue
        
        base_key, sample_type, idx = parse_key(key)
        
        if base_key and sample_type and idx is not None:
            base_keys.add(base_key)
            grouped[base_key][idx][sample_type] = conversation
            grouped[base_key][idx][f'{sample_type}_key'] = key
    
    import pdb; pdb.set_trace()

    base_keys_list = sorted(list(base_keys))  # Sort for reproducibility
    random.shuffle(base_keys_list)

    split_idx = len(base_keys_list) - 100
    train_keys = set(base_keys_list[:split_idx])
    valid_keys = set(base_keys_list[split_idx:])

    # Create separate grouped dictionaries for train and valid
    grouped_train = defaultdict(lambda: defaultdict(dict))
    grouped_valid = defaultdict(lambda: defaultdict(dict))

    # Assign to train or valid
    for base_key in base_keys:
        if base_key in train_keys:
            grouped_train[base_key] = grouped[base_key]
        elif base_key in valid_keys:
            grouped_valid[base_key] = grouped[base_key]
    
    # Process training set
    train_dataset = DialogueDataset(task="audio_dialogue")
    
    for base_key, indices_dict in grouped_train.items():
        for idx, samples in indices_dict.items():
            # Skip if we don't have both positive and negative
            if 'positive' not in samples or 'negative' not in samples:
                continue
            
            combined = []
            # codec_paths = []
            
            # Extract from positive sample
            user_pos, assistant_pos, total_pos = extract_last_text_bpe(samples['positive'])
            if user_pos and assistant_pos:
                combined.append(user_pos)
                combined.append(assistant_pos)
                pos_key = samples['positive_key']
                # codec_paths.append(codec_mapping[pos_key])
            
            # Extract from negative sample
            user_neg, assistant_neg, total_neg = extract_last_text_bpe(samples['negative'])
            if user_neg and assistant_neg:
                combined.append(user_neg)
                combined.append(assistant_neg)
                neg_key = samples['negative_key']
                # codec_paths.append(codec_mapping[neg_key])
            
            assert total_pos == total_neg
            
            if combined:
                assert len(combined) == 4
                dialogue = Dialogue(task="audio_dialogue")
                
                # Add context
                for k in total_pos:
                    dialogue.add_segment(
                        role=k[0],
                        modality=k[1],
                        target=False,
                        content=k[-1],
                    )
                
                # Add positive pair
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
                # dialogue.add_segment(
                #     role="assistant",
                #     modality='codec_ssl',
                #     target=False,
                #     content=codec_paths[0],
                # )
                
                # Add context again
                for k in total_pos:
                    dialogue.add_segment(
                        role=k[0],
                        modality=k[1],
                        target=False,
                        content=k[-1],
                    )
                
                # Add negative pair
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
                # dialogue.add_segment(
                #     role="assistant",
                #     modality='codec_ssl',
                #     target=False,
                #     content=codec_paths[1],
                # )
                
                # Create unique example_id with index
                example_id = f"{base_key}_{idx}"
                train_dataset.add_dialogue(example_id, dialogue)
    
    # Output training result
    output_dir = Path(output_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset.dump_dataset(output_dir)
    
    # Process validation set
    valid_dataset = DialogueDataset(task="audio_dialogue")
    
    for base_key, indices_dict in grouped_valid.items():
        for idx, samples in indices_dict.items():
            # Skip if we don't have both positive and negative
            if 'positive' not in samples or 'negative' not in samples:
                continue
            
            combined = []
            # codec_paths = []
            
            # Extract from positive sample
            user_pos, assistant_pos, total_pos = extract_last_text_bpe(samples['positive'])
            if user_pos and assistant_pos:
                combined.append(user_pos)
                combined.append(assistant_pos)
                pos_key = samples['positive_key']
                # codec_paths.append(codec_mapping[pos_key])
            
            # Extract from negative sample
            user_neg, assistant_neg, total_neg = extract_last_text_bpe(samples['negative'])
            if user_neg and assistant_neg:
                combined.append(user_neg)
                combined.append(assistant_neg)
                neg_key = samples['negative_key']
                # codec_paths.append(codec_mapping[neg_key])
            
            assert total_pos == total_neg
            
            if combined:
                assert len(combined) == 4
                dialogue = Dialogue(task="audio_dialogue")
                
                # Add context
                for k in total_pos:
                    dialogue.add_segment(
                        role=k[0],
                        modality=k[1],
                        target=False,
                        content=k[-1],
                    )
                
                # Add positive pair
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
                # dialogue.add_segment(
                #     role="assistant",
                #     modality='codec_ssl',
                #     target=False,
                #     content=codec_paths[0],
                # )
                
                # Add context again
                for k in total_pos:
                    dialogue.add_segment(
                        role=k[0],
                        modality=k[1],
                        target=False,
                        content=k[-1],
                    )
                
                # Add negative pair
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
                # dialogue.add_segment(
                #     role="assistant",
                #     modality='codec_ssl',
                #     target=False,
                #     content=codec_paths[1],
                # )
                
                # Create unique example_id with index
                example_id = f"{base_key}_{idx}"
                valid_dataset.add_dialogue(example_id, dialogue)
    
    # Output validation result
    output_valid_dir = Path(output_valid_file)
    output_valid_dir.mkdir(parents=True, exist_ok=True)
    valid_dataset.dump_dataset(output_valid_dir)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    
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
