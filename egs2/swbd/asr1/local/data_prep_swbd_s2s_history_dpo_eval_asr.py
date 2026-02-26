#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from pathlib import Path
import random

# Set seed for reproducibility
random.seed(42)
from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset

def extract_last_text_bpe(conversation):
    """Extract the last user and assistant text_bpe entries from a conversation."""
    last_user = None
    last_assistant = None
    
    # Iterate through conversation to find last user and assistant text_bpe entries
    total_entry=[]
    for entry in conversation:
        total_entry.append(entry)
        if entry[0] == "user":
            last_user = entry
        elif entry[0] == "assistant":
            last_assistant = entry
    
    return last_user, last_assistant, total_entry[:-3]

def process_file(input_file, output_file=None, output_valid_file=None):
    """Process the input JSON file and extract text_bpe entries."""

    # Read input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Process each group
    
    test_dataset = DialogueDataset(task="audio_dialogue")
    for key, conversation in data.items():
        
        # Extract from positive sample
        user_pos, assistant_pos, total_pos = extract_last_text_bpe(conversation)
        
        
        dialogue = Dialogue(task="audio_dialogue")
        for k in total_pos:
            dialogue.add_segment(
                role=k[0],
                modality=k[1],
                target=False,
                content=k[-1],
            )
        dialogue.add_segment(
            role=user_pos[0],
            modality='text_bpe',
            target=True,
            content=user_pos[-1],
        )
        example_id = key
        test_dataset.add_dialogue(example_id, dialogue)
    
    # Output result
    output_dir=Path(output_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_dataset.dump_dataset(output_dir)
    
    return

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_text_bpe.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    process_file(input_file, output_dir)