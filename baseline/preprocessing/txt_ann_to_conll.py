import os
import re
from nltk.tokenize import sent_tokenize, word_tokenize

def parse_ann_file(ann_file):
    """
    Parse the .ann file and extract annotation information.
    Returns a list of tuples (start, end, entity, text).
    """
    annotations = []
    with open(ann_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            tag_info, text = parts[1], parts[2]
            tag_parts = tag_info.split()
            if len(tag_parts) < 3:
                continue

            entity = tag_parts[0]
            start = int(tag_parts[1])
            end = int(tag_parts[-1])

            annotations.append((start, end, entity, text))

    return annotations

def convert_to_conll(txt_file, ann_file):
    """
    Convert a .txt and its corresponding .ann file to CoNLL format.
    Returns a list of CoNLL-formatted lines.
    """
    with open(txt_file, 'r', encoding='utf-8') as file:
        text = file.read()

    annotations = parse_ann_file(ann_file)
    token_lines = []

    # Create a map of character offsets to tags
    tags = ['O'] * len(text)
    for start, end, entity, _ in annotations:
        for i in range(start, end):
            if i == start:
                tags[i] = f'B-{entity}' if entity != 'MajorClaim' else 'B-Claim'
            else:
                tags[i] = f'I-{entity}' if entity != 'MajorClaim' else 'I-Claim'

    # Sentence tokenization
    sentences = sent_tokenize(text)
    offset = 0
    token_id = 0

    for sentence in sentences:
        words = word_tokenize(sentence)
        for word in words:
            word_start = text.find(word, offset)
            word_end = word_start + len(word)

            if word_start != -1:
                tag = tags[word_start] if word_start < len(tags) else 'O'
                token_lines.append((token_id, word, '_', '_', tag))
                token_id += 1
            offset = word_end

        token_lines.append(('', '', '', '', ''))  # Sentence boundary

    # Convert to CoNLL format
    conll_lines = []
    for token_info in token_lines:
        if token_info[1]:
            conll_lines.append(f"{token_info[0]}\t{token_info[1]}\t{token_info[2]}\t{token_info[3]}\t{token_info[4]}")
        else:
            conll_lines.append("")

    return conll_lines

def process_folder(master_folder):
    """
    Process all .txt and .ann files in the master folder and convert them to a CoNLL file.
    """
    folder_name = os.path.basename(os.path.normpath(master_folder))
    output_file = os.path.join(os.getcwd(), f"{folder_name}.conll")

    conll_lines = []

    for filename in os.listdir(master_folder):
        if filename.endswith('.txt'):
            txt_file = os.path.join(master_folder, filename)
            ann_file = os.path.splitext(txt_file)[0] + '.ann'

            if os.path.exists(ann_file):
                conll_lines.extend(convert_to_conll(txt_file, ann_file))
                conll_lines.append("")  # Separate files with a blank line

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(conll_lines))

    print(f"CoNLL file saved as {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <master_folder>")
        sys.exit(1)

    master_folder = sys.argv[1]

    if os.path.isdir(master_folder):
        process_folder(master_folder)
    else:
        print("The specified folder does not exist.")
