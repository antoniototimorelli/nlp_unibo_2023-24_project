import os
from rapidfuzz import fuzz
import csv

# searchset = "neoplasm"
# searchset = "glaucoma"
# searchset = "mixed"

# parent_folder = f"../../../data/test/{searchset}_test"
# search_phrases_file = f"../data/{searchset}_test/seqtag_decoded_texts.txt"
# output_tsv = f"../data/{searchset}_test/seqtag_results.tsv"

parent_folder = f"../../../data/custom_datasets/test"
search_phrases_file = f"../data/custom_datasets/test/seqtag_decoded_texts.txt"
output_tsv = f"../data/custom_datasets/test/seqtag_results.tsv"


def load_search_phrases(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    # Split phrases based on double newlines (blank lines)
    phrases = [phrase.strip() for phrase in content.split("\n\n") if phrase.strip() != ""]
    
    # Split each phrase into a tuple by the first newline
    phrases_tuples = [tuple(phrase.split("\n", 1)) for phrase in phrases]

    return phrases_tuples

# Load search phrases
search_phrases = load_search_phrases(search_phrases_file)

# Counter for found phrases
found_phrases = list()

# Open the TSV file for writing results
with open(output_tsv, "w", newline="", encoding="utf-8") as tsv_file:
    tsv_writer = csv.writer(tsv_file, delimiter="\t")
    # Write the header row
    tsv_writer.writerow(["Search Phrase", "Filename", "Type"])

    for typo, phrase in search_phrases:
        highest_similarity = 0
        best_file = None

        print(f"\nSearching for phrase: {phrase}")
        
        # Iterate over files in the parent folder
        for filename in os.listdir(parent_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(parent_folder, filename)
                #print(f"  Checking file: {file_path}")

                # Open the current file and read its entire content
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()

                # Check for matches for the search phrase in the file content
                similarity = fuzz.partial_ratio(phrase.lower(), content.lower())
                #print(f"    Similarity with {filename}: {similarity}%")

                # Update highest similarity and best file if this is the best match so far
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_file = filename
                    

        # Write the best result to the TSV file if it meets the threshold
        print(f"  Best match for phrase:")
        print(f"    File: {best_file}")
        print(f"    Similarity: {highest_similarity}%")
        tsv_writer.writerow([phrase, os.path.splitext(best_file)[0], typo])
        found_phrases.append((typo, phrase))

# Determine phrases not found
not_found_phrases = [phrase for phrase in search_phrases if phrase not in found_phrases]

# Print summary
print("\nSearch Summary:")
print(f"Total phrases searched: {len(search_phrases)}")
print(f"Total phrases found: {len(found_phrases)}")
print(f"Total phrases not found: {len(not_found_phrases)}")
if not_found_phrases:
    print("\nPhrases not found:")
    for phrase in not_found_phrases:
        print(f"  - {phrase}")
