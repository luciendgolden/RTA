# File: baby_llama_clean.py
# -------------------------
# Data cleaning script, taken from the BabyLlama repository (https://github.com/timinar/BabyLlama)
# of Timirsayov and Tastet, 2023.

import argparse
import os
import re
from pathlib import Path
import spacy

nlp = spacy.blank("en")
nlp.max_length = 10**7

def _make_padding_sequence(seq_length):
    return ''

def cleanup_extra_spaces(text):
    multiple_spaces_ex = re.compile(r'[ \t\u00A0]+')
    space_before_punctuation_ex = re.compile(r'[ \t\u00A0]([.,;!?])')
    text = multiple_spaces_ex.sub(' ', text)
    text = space_before_punctuation_ex.sub(r'\1', text)
    return text

def post_process_for_glove(text):
    processed_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        doc = nlp(line)
        tokens = [token.text.lower() for token in doc]
        if tokens:
            processed_lines.append(' '.join(tokens))
    return '\n'.join(processed_lines)

# BNC SPOKEN
def cleanup_bnc_spoken(text, seq_length):
    pad_seq = _make_padding_sequence(seq_length)
    text = cleanup_extra_spaces(text)
    text = re.sub(r'\n\n', pad_seq, text) + pad_seq
    return text

# CHILDES
def cleanup_aochildes(text, seq_length):
    text = cleanup_extra_spaces(text)
    text = re.sub(r'\n\n', '\n', text)
    return text + _make_padding_sequence(seq_length)

# GUTENBERG
def cleanup_gutenberg(text, seq_length):
    # Overall, the text is clean, however some entries don’t seem
    # very useful, e.g. figure captions preceded by a number.
    # Not sure if we should remove them, because that would also
    # remove bullet lists which are otherwise consistent with the
    # surrounding text.
    # No start or end tokens because the text seems to be cut.
    text = cleanup_extra_spaces(text)
    text = re.sub(r'\n\n', '\n', text)
    return text

# OPEN SUBTITLES
def cleanup_open_subtitles(text, seq_length):
    # The text is mostly clean, apart from some subtitle credits
    # such as "Subtitles by ...".
    subtitle_credit_ex = re.compile(r'^.*subtitle.*$\n', re.MULTILINE | re.IGNORECASE)
    text = subtitle_credit_ex.sub('', text)
    text = cleanup_extra_spaces(text)
    text = re.sub(r'\n\n', '\n', text)
    return text + _make_padding_sequence(seq_length)

# SIMPLE WIKI
def cleanup_simple_wikipedia(text, seq_length):
    pad_seq = _make_padding_sequence(seq_length)
    text = cleanup_extra_spaces(text)
    text = re.sub(r'\n\n', '\n', text)
    text = text.lstrip('\n')
    return text

# SWITCHBOARD
def cleanup_switchboard(text, seq_length):
    # No start or end tokens because the text seems to be cut.
    text = cleanup_extra_spaces(text)
    text = re.sub(r'\n\n', '\n', text)
    return text

CLEANUP_FUNCTIONS = {
    'bnc_spoken': cleanup_bnc_spoken,
    'childes': cleanup_aochildes,
    'gutenberg': cleanup_gutenberg,
    'open_subtitles': cleanup_open_subtitles,
    'simple_wiki': cleanup_simple_wikipedia,
    'switchboard': cleanup_switchboard,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean BabyLM data files in a specified input directory")
    parser.add_argument("--input_dir", required=True, help="Directory containing the files to clean")
    parser.add_argument("--output_dir", required=True, help="Directory to save cleaned files")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length for padding (default: 512)")
    args = parser.parse_args()
    
    INPUT_DIR = Path(args.input_dir)
    OUTPUT_ROOT = Path(args.output_dir)
    
    all_files = [file for file in INPUT_DIR.iterdir() if file.is_file() and file.suffix in ['.train', '.dev', '.test']]

    for file in all_files:
        text = file.read_text()
        corpus_name = file.stem

        if corpus_name in CLEANUP_FUNCTIONS:
            print(f"Cleaning {corpus_name}...")
            cleaned_text = CLEANUP_FUNCTIONS[corpus_name](text, args.seq_length)
        else:
            print(f"No cleaning function for {corpus_name}, skipping...")
            continue
        
        print(f"Tokenizing {corpus_name}...")
        cleaned_text = post_process_for_glove(cleaned_text)
        
        new_file = OUTPUT_ROOT / file.name
        os.makedirs(os.path.dirname(new_file), exist_ok=True)

        with open(new_file, 'w') as f:
            f.write(cleaned_text)
        print(f"🖖 Tokenized '{file.name}' (size {len(text)} -> {len(cleaned_text)})")