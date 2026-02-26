#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from pathlib import Path
import random

# Set seed for reproducibility
random.seed(42)
from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset




def process_file(input_file, output_file=None, output_valid_file=None):
    """Process the input JSON file and extract text_bpe entries."""

    # Read input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Group by base key (without _positive/_negative suffix)
    
    
    train_dataset = DialogueDataset(task="audio_dialogue")
    for key, conversation in data.items():
        dialogue = Dialogue(task="audio_dialogue")
        for k in conversation[:-1]:
            dialogue.add_segment(
                role=k[0],
                modality=k[1],
                target=k[2],
                content=k[-1],
            )
        example_id = key
        train_dataset.add_dialogue(example_id, dialogue)
    
    # Output result
    output_dir=Path(output_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset.dump_dataset(output_dir)
    
    
    
    return

if __name__ == "__main__":
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    process_file(input_file, output_dir)