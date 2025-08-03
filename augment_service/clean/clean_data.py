# https://huggingface.co/datasets/cambridge-climb/BabyLM/blob/main/clean_data.py
""" Script used to clean the data. """

import os
import re

from augment_service.config.settings import BABYLM_FOLDERS, BABYLM_ORIGINAL_DATA_PATH

def clean_bnc_spoken(lines):
    """ For bnc_spoken, we only remove empty lines """
    new_lines = []
    for line in lines:
        if line != '\n':
            new_lines.append(line)
    return new_lines

def clean_childes(lines):
    """ For childes, we remove the space between the punctuation mark and the final word and join together every 5 lines """
    new_lines = []
    for line in lines:
        new_line = line.strip()
        new_lines.append(new_line + '\n')
    return new_lines

def clean_gutenberg(lines):
    """ For gutenberg we lowercase, extract narrative content, remove metadata, headings, and artifacts, and normalize text """
    paragraphs = []
    paragraph = ""
    skip_metadata = True
    in_footer = False
    narrative_keywords = ["the ", "a ", "in ", "he ", "she ", "they ", "it "]
    metadata_indicators = ["project gutenberg", "title:", "author:", "release date:", "language:", "start of"]
    footer_indicators = ["end of the project gutenberg", "end of this project gutenberg", "*** end"]
    heading_indicators = ["chapter ", "section ", "part "]

    for line in lines:
        # Preprocess: strip, remove italics, normalize apostrophes
        tmp_line = line.strip().replace('_', '').replace('’', "'")

        # skip empty lines
        if not tmp_line:
            if paragraph and not skip_metadata and not in_footer:
                if (len(paragraph.split()) > 5 and
                    not paragraph.split()[-1][-1].isdigit() and
                    not any(indicator in paragraph for indicator in heading_indicators)):
                    paragraphs.append(paragraph[:-1])
                paragraph = ""
            continue

        # skip metadata
        if skip_metadata:
            if any(indicator in tmp_line.lower() for indicator in metadata_indicators):
                continue
            if any(tmp_line.startswith(keyword) for keyword in narrative_keywords) or not any(indicator in tmp_line for indicator in heading_indicators):
                skip_metadata = False
            else:
                continue

        # detect footer
        if any(indicator in tmp_line for indicator in footer_indicators):
            in_footer = True
            continue
        if in_footer:
            continue

        # skip standalone headings
        if any(indicator in tmp_line for indicator in heading_indicators) or tmp_line.startswith('*'):
            if paragraph:  # Append accumulated paragraph before heading
                if (len(paragraph.split()) > 5 and
                    not paragraph.split()[-1][-1].isdigit() and
                    not any(indicator in paragraph for indicator in heading_indicators)):
                    paragraphs.append(paragraph[:-1])
                paragraph = ""
            continue

        # each non-heading line as a paragraph
        if paragraph:
            if (len(paragraph.split()) > 5 and
                not paragraph.split()[-1][-1].isdigit() and
                not any(indicator in paragraph for indicator in heading_indicators)):
                paragraphs.append(paragraph[:-1])
            paragraph = ""
        paragraph = tmp_line + " "

    # handle final paragraph
    if paragraph and not skip_metadata and not in_footer:
        if (len(paragraph.split()) > 5 and
            not paragraph.split()[-1][-1].isdigit() and
            not any(indicator in paragraph for indicator in heading_indicators)):
            paragraphs.append(paragraph[:-1])

    # filtering
    bad_chars = ['*', 'p.', '=', '|', '[', ']', '       ', '    ', 'v.', '---', '--']
    bad_patterns = [
        r'\[\d+\]',  # footnotes
        r'project gutenberg', r'transcriber\'s note', r'end of.*gutenberg',
        r'^\d+$'  # page numbers
    ]

    new_lines = []
    for p in paragraphs:
        if (not any(c in p for c in bad_chars) and
            not any(re.search(pattern, p) for pattern in bad_patterns) and
            p.strip() and p[0] != '(' and len(p.strip()) > 10):
            p = re.sub(r'\s+', ' ', p.strip())
            new_lines.append(p + '\n')

    return new_lines

def clean_open_subtitles(lines):
    """ For open_subtitles, we lowercase, remove subtitle dashes and fix the lowercase 'l' problem. We also join every 5 lines. """
    punctuation = ['.', ',', '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '"', "'", '“', '”', '—', '–', ' ', '\n']
    new_lines = []
    joined = []
    count = 0
    for line in lines:
        new_line = line
        # Skip music lines
        if '♪' in new_line or '[' in new_line or ']' in new_line or '‎' in new_line:
            continue
        if new_line[0:2] in ["- ", "– ", "— "]:
            new_line = new_line[2:]
        if new_line[0] in ["-", "–", "—"]:
            new_line = new_line[1:]
        new_line = ' ' + new_line
        for punct in punctuation:
            new_line = new_line.replace(f" l{punct}", f" i{punct}")
            new_line = new_line.replace(f" lm{punct}", f" im{punct}")
            new_line = new_line.replace(f" lf{punct}", f" if{punct}")
        new_line = new_line.replace(' lc', ' ic')
        new_line = new_line.replace(' ld', ' id')
        new_line = new_line.replace(' lj', ' i j')
        new_line = new_line.replace(' ln', ' in')
        new_line = new_line.replace(' lp', ' ip')
        new_line = new_line.replace(' lr', ' ir')
        new_line = new_line.replace(' ls', ' is')
        new_line = new_line.replace(' isd', ' lsd')
        new_line = new_line.replace(' lt', ' it')
        new_line = new_line.replace(' lt', ' it')
        new_line = new_line.replace(' lv', ' iv')
        if new_line.strip() != '':
            joined.append(new_line.strip())
            count += 1
            if count % 5 == 0:
                new_lines.append(" ".join(joined) + '\n')
                joined = []
    return new_lines

def clean_simple_wikipedia(lines):
    """ For wikipedia, we lowercase and remove empty lines and article names.
     We also remove lines that seem to be figure names or table entries. """
    new_lines = []
    for line in lines:
        new_line = line.strip()
        words = new_line.split()
        
        # Remove empty lines and article names
        if new_line == "":
            continue
        if new_line[0] == "=" and new_line[-1] == "=":
            continue

        # Filter out lines that seem to be figure names or table entries
        all_numeric = True
        all_uppercase = True
        for word in words:
            if not word.isnumeric():
                all_numeric = False
            if not word[0].isupper():
                all_uppercase = False
        if all_numeric or all_uppercase:
            continue
    
        new_lines.append(new_line.strip() + '\n')
    return new_lines

def clean_switchboard(lines):
    """ For switchboard, we join every 5 lines. """
    new_lines = []
    count = 0
    joined = []
    for line in lines:
        new_line = line.strip()
        joined.append(new_line)
        count += 1
        if count % 5 == 0:
            new_lines.append(" ".join(joined) + '\n')
            joined = []
    return new_lines

CLEAN_FUNCTIONS = {
    'bnc_spoken': clean_bnc_spoken,
    'childes': clean_childes,
    'gutenberg': clean_gutenberg,
    'open_subtitles': clean_open_subtitles,
    'simple_wiki': clean_simple_wikipedia,
    'switchboard': clean_switchboard
}

if __name__ == "__main__":
    # Read all text files from directory "BabyLM"
    all_files = []
    for folder in BABYLM_FOLDERS:
        for root, dirs, files in os.walk(f"{BABYLM_ORIGINAL_DATA_PATH}/{folder}"):
            for file in files:
                # if file ends with .dev, .test, .train or .txt
                if file.endswith(('.dev', '.test', '.train')):
                    all_files.append(os.path.join(root, file))

    for file in all_files:
        with open(file, 'r') as f:
            lines = f.readlines()

        # Get the corpus name
        corpus_name = os.path.basename(file).split('.')[0]

        # Clean the data
        if corpus_name in CLEAN_FUNCTIONS:
            print(f"Cleaning {corpus_name}...")
            #print(lines[:5])
            lines = CLEAN_FUNCTIONS[corpus_name](lines)
            #print(f"Cleaned lines for {file}:")
            #print(lines[:5])
            lines = [re.sub(r'\s+', ' ', line.rstrip()) + '\n' for line in lines if line.strip() != '']
            #print(lines[:5])
        else:
            continue
            

        # Determine the folder (10M, 100M, dev, test)
        folder = None
        path_parts = file.split(os.sep)
        original_idx = path_parts.index('original')
        if original_idx + 1 < len(path_parts):
            folder = path_parts[original_idx + 1]
            if folder not in BABYLM_FOLDERS:
                folder = None
            
        # Write the new file
        new_file = file.replace('original', 'clean')
        print(new_file)
        os.makedirs(os.path.dirname(new_file), exist_ok=True)
        with open(new_file, 'w') as f:
           f.writelines(lines)
