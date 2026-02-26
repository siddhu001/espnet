import re
from difflib import SequenceMatcher
import sys

def similarity_ratio(s1, s2):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

def remove_repetitive_sentences(text, similarity_threshold=0.85):
    """Remove exact and near-duplicate sentence repetitions"""
    # Split by sentence-ending punctuation
    sentences = re.split(r'([.!?]+)', text.strip())
    
    # Reconstruct sentences with their punctuation
    reconstructed = []
    for i in range(0, len(sentences)-1, 2):
        if sentences[i].strip():
            sent = sentences[i].strip()
            punct = sentences[i+1] if i+1 < len(sentences) else '.'
            reconstructed.append((sent, punct))
    
    # Handle last sentence if no punctuation
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        reconstructed.append((sentences[-1].strip(), '.'))
    
    # Remove duplicates and near-duplicates
    unique = []
    for sent, punct in reconstructed:
        is_duplicate = False
        for prev_sent, _ in unique:
            if similarity_ratio(sent, prev_sent) > similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append((sent, punct))
    
    # Reconstruct text
    result = ' '.join(sent + punct for sent, punct in unique)
    return result.strip()

def remove_trailing_repetition(text):
    """Remove trailing repeated sentences"""
    # More aggressive pattern matching
    # Pattern: "X. X." where X can have slight variations
    sentences = re.split(r'[.!?]+\s*', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) >= 2:
        # Check if last two sentences are similar
        if similarity_ratio(sentences[-1], sentences[-2]) > 0.85:
            sentences = sentences[:-1]
    
    return '. '.join(sentences) + '.' if sentences else text

def truncate_incomplete(text):
    """Remove incomplete trailing fragments"""
    # Remove trailing text that doesn't end with proper punctuation
    text = text.strip()
    if text and text[-1] not in '.!?':
        # Find last proper sentence ending
        last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_punct > 0:
            text = text[:last_punct+1]
    return text

def postprocess_output(text):
    """Apply all postprocessing steps"""
    if not text or len(text.strip()) == 0:
        return text
    
    # Step 1: Remove incomplete fragments
    text = truncate_incomplete(text)
    
    # Step 2: Remove sentence repetitions (exact and near-duplicates)
    text = remove_repetitive_sentences(text)
    
    # Step 3: Remove trailing repetitions (backup)
    text = remove_trailing_repetition(text)
    
    # Step 4: Clean up spacing and punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+([.!?,])', r'\1', text)
    text = re.sub(r'\.+', '.', text)  # Remove multiple periods
    text = re.sub(r'[,;:]+([.!?])', r'\1', text)
    
    return text

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Read input file
    outputs = []
    ids = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                ids.append(parts[0])
                outputs.append(parts[1])
            elif len(parts) == 1:
                # Handle lines with ID but no text
                ids.append(parts[0])
                outputs.append("")
    
    print(f"Read {len(outputs)} lines from {input_file}")
    
    # Process outputs
    cleaned_outputs = [postprocess_output(text) for text in outputs]
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for id_, cleaned in zip(ids, cleaned_outputs):
            f.write(f"{id_} {cleaned}\n")
    
    print(f"Wrote {len(cleaned_outputs)} cleaned lines to {output_file}")
    
    # Print first few examples for verification
    print("\nFirst 5 examples:")
    for i in range(min(5, len(outputs))):
        print(f"\n{i+1}. ID: {ids[i]}")
        print(f"   Original: {outputs[i]}")
        print(f"   Cleaned:  {cleaned_outputs[i]}")