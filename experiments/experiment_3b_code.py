# Standard library imports
import os
import shutil
import random
import optuna
import time
from pathlib import Path
from itertools import combinations, pairwise
from collections import defaultdict, Counter
from copy import copy

# Third-party library imports
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import textwrap
import optuna
from tqdm import tqdm
from tabulate import tabulate

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Scikit-learn imports
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as tF
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.nn import MSELoss

# QuaPy imports
import quapy as qp
import quapy.functional as F
from quapy.data import Dataset
from quapy.method.meta import QuaNet
from quapy.classification.neural import NeuralClassifierTrainer, CNNnet
from quapy.error import ae, rae, mse, mae, mrae, mkld
from quapy.util import EarlyStop
from quapy.protocol import UPP
from quapy.method.aggregative import *

nltk.download('punkt')
nltk.download('punkt_tab')

DEBUG = True
LEVEL = 1 # ONLY PRINT
# LEVEL = 2 # ONLY  FILE
# LEVEL = 3 # BOTH

def debug_print(*args, filename: str = None, **kwargs):
    """
    Prints according to debug level

    Parameters:
        args (str): What is printed.
        filename (str): File where the output is stored if level is either 2 or 3.
    """
    if DEBUG:
        # Print to console
        if LEVEL == 1 or LEVEL == 3:
            print(*args, **kwargs)
        
        # If a filename is provided, write to that file
        if LEVEL == 2 or LEVEL == 3:
            if filename:
                with open(filename, "a") as file:  # Open the file in append mode
                    print(*args, file=file, **kwargs)

def set_seed(seed: int):
    """
    Sets the seed for reproducibility in random, numpy, and PyTorch libraries.

    Parameters:
        seed (int): The seed value to ensure reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# COMPONENTS DATASET

def split_by_boundaries_components(text: str, components_boundaries: dict, components_types: dict, positives: list):
    """
    Splits the text into labeled parts based on component boundaries and their types.

    Parameters:
        text (str): The original text to split.
        components_boundaries (dict): A dictionary mapping component IDs to their boundaries (start and end positions).
        components_types (dict): A dictionary mapping component IDs to their types (e.g., 'Claim', 'Premise').
        positives (list): A list of component types considered as "positive" labels (e.g., 'Claim', 'Premise').

    Returns:
        list: A list of tuples (text_part, label), where label is:
            - 0 -> Negative class
            - 1 -> Positive class (Either Claims, Premise or both)
    """
    labeled_parts = []
    last_pos = 0

    # Iterate over sorted boundaries and extract the corresponding text parts
    sorted_boundaries = sorted([(start, end, components_types[key]) 
                                for key, bounds in components_boundaries.items() 
                                for start, end in bounds])

    for start, end, label_type in sorted_boundaries:
        # Add the non-labeled part between the previous boundary and the current one
        if last_pos < start:
            non_labeled_part = text[last_pos:start]
            if non_labeled_part.strip():  # Avoid empty parts
                labeled_parts.append((non_labeled_part, 0))

        # Add the labeled part (Claim or Premise)
        labeled_parts.append((text[start:end], 1 if label_type in positives else 0))
        
        last_pos = end

    # Add the remaining part of the text (after the last component)
    if last_pos < len(text):
        remaining_part = text[last_pos:]
        if remaining_part.strip():
            labeled_parts.append((remaining_part, 0))

    return labeled_parts

def label_sentences_components(text: str, components_boundaries: dict, components_types: dict, positives: list):
    """
    Labels the text into sentences by splitting it based on component boundaries and tokenizing remaining parts.

    Parameters:
        text (str): The original text to process.
        components_boundaries (dict): Mapping of component IDs to boundaries (start, end positions).
        components_types (dict): Mapping of component IDs to their types.
        positives (list): A list of component types considered as "positive" labels.

    Returns:
        list: A list of dictionaries, where each entry contains:
            - 'sentence' (str): A sentence from the text.
            - 'label' (int): The label for the sentence (0 for None, 1 for Claim/Premise).
    """
    labeled_sentences = []
    
    # Split text based on boundaries and label Claims/Premises
    labeled_parts = split_by_boundaries_components(text, components_boundaries, components_types, positives)
    
    # Now process each part
    for part, label in labeled_parts:
        if label == 0:
            # For the non-labeled parts, split into sentences
            sentences = sent_tokenize(part)
            for sentence in sentences:
                labeled_sentences.append({'sentence': sentence, 'label': 0})
        else:
            # For labeled parts (Claim or Premise), treat the entire part as one sentence
            labeled_sentences.append({'sentence': part, 'label': label})

    return labeled_sentences

def read_brat_dataset_components(folder: str, positives: list = ['Claim', 'MajorClaim', 'Premise']):
    """
    Reads BRAT dataset annotations and extracts labeled sentences.

    Parameters:
        folder (str): Path to the folder containing .ann and .txt files.
        positives (list): List of component types to label as positive (default: ['Claim', 'MajorClaim', 'Premise']).

    Returns:
        list: A dataset where each entry is a dictionary with:
            - 'text' (str): The full text of the document.
            - 'filename' (str): The name of the file.
            - 'sentence' (str): The labeled sentence.
            - 'label' (int): The label for the sentence (0 or 1).
    """
    dataset, temp_sentence, merge = [], "", False
    
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.ann'):
                ann_path = os.path.join(root, file)
                txt_path = ann_path.replace('.ann', '.txt')
                
                if os.path.exists(txt_path):
                    with open(ann_path, 'r', encoding='utf-8') as ann_f, open(txt_path, 'r', encoding='utf-8') as txt_f:
                        annotations = [line.strip().split('\t') for line in ann_f]
                        text = txt_f.read()

                        components_boundaries = {
                            ann[0]: [(int(ann[1].split(' ')[1]), int(ann[1].split(' ')[2]))] 
                            for ann in annotations if ann[0].startswith('T')
                        }
                        components_types = {
                            ann[0]: ann[1].split(' ')[0] 
                            for ann in annotations if ann[0].startswith('T')
                        }

                        # Label sentences based on boundaries first, then split the remaining text into sentences
                        labeled_sentences = label_sentences_components(text, components_boundaries, components_types, positives)
                        
                        for sentence in labeled_sentences:
                            if len(sentence['sentence'].strip().split()) < 3 and sentence['label'] == 0:
                                # print(sentence['sentence'])
                                temp_sentence += sentence['sentence'].strip() + " "
                                merge = True
                            else:
                                if merge:
                                    temp_sentence += sentence['sentence'].strip()
                                    merge = False
                                else:
                                    temp_sentence = sentence['sentence'].strip()

                                dataset.append({
                                    'text': text,
                                    'filename': file.split('.')[0], 
                                    'sentence': temp_sentence,
                                    'label': sentence['label']  
                                })

                                temp_sentence = ""
    
    return dataset

def compute_dataset_statistics_components(dataset: list, dataset_name: str = "dataset", label_for_component: int = 1, label_for_non_component: int = 0):
    """
    Computes and displays statistics for a labeled components dataset.

    Parameters:
        dataset (list): A dataset where each element contains 'sentence', 'label', and 'filename'.
        dataset_name (str): The name of the dataset (e.g., 'train', 'test') for display (default: "dataset").
        label_for_component (int): Label representing a "component" in the dataset (default: 1).
        label_for_non_component (int): Label representing a "non-component" in the dataset (default: 0).

    Returns:
        tuple: Contains:
            - label_counts (dict): A dictionary with counts for each label.
            - avg_sentences_per_file (float): The average number of sentences per file.
    """
    print(f'- {dataset_name.capitalize()} set:')
    max_len, max_sentences = 0, 0
    label_counts = {}
    
    # Initialize counts for components and non-components per file
    filename_to_components = defaultdict(int)
    filename_to_non_components = defaultdict(int)
    
    # Iterate over the dataset
    for data in dataset:
        # Update max sentence length
        sentence_len = len(data['sentence'].strip().split(" "))
        if sentence_len > max_len:
            max_len = sentence_len

        # Count the labels
        label = data['label']
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

        # Count components and non-components per filename
        filename = data['filename']
        if label == label_for_component:
            filename_to_components[filename] += 1
        elif label == label_for_non_component:
            filename_to_non_components[filename] += 1

    for filename in filename_to_components.keys():
        n_sentences = filename_to_components[filename] + filename_to_non_components[filename]
        if n_sentences > max_sentences:
            max_sentences = n_sentences

    # Print label counts
    for label, count in sorted(label_counts.items()):
        print(f'\tLabel {label}: {count} samples')

    # Calculate average number of sentences, components, and non-components per file
    avg_sentences_per_file = round(pd.DataFrame(dataset).groupby('filename').size().mean())
    avg_components_per_file = sum(filename_to_components.values()) / len(filename_to_components)
    avg_non_components_per_file = sum(filename_to_non_components.values()) / len(filename_to_non_components)

    # Display statistics
    print(f'\n\tThere are {len(label_counts)} different labels in the {dataset_name} set -> {sorted(list(label_counts.keys()))}')
    print(f'\tAverage number of sentences per file in {dataset_name} set: {avg_sentences_per_file}')
    print(f'\tMax sentence length: {max_len}')
    print(f'\tMax sentences in a single abstract: {max_sentences}')
    print(f'\tAverage components per file: {avg_components_per_file:.2f}')
    print(f'\tAverage non-components per file: {avg_non_components_per_file:.2f}\n')

    return label_counts, avg_sentences_per_file

# RELATIONSHIPS DATASET

def split_by_boundaries_relations(text: str, components_boundaries: dict, components_types: dict, relations: list):
    """
    Splits the text into labeled parts based on component boundaries and wether they are first argument of a relation.

    Parameters:
        text (str): The original text to split.
        components_boundaries (dict): A dictionary mapping component IDs to their boundaries (start and end positions).
        components_types (dict): A dictionary mapping component IDs to their types (e.g., 'Claim', 'Premise').
        relations (list): A list of abstract's relations between argumentative components.

    Returns:
        list: A list of tuples (text_part, label), where label is:
            - 0 -> Negative class
            - 1 -> Positive class (Component is first argument of a relation)
    """
    labeled_parts = []
    last_pos = 0

    # Determine which components are the first arguments of a relation
    arg1_set = set([rel['arg1_id'] for rel in relations])

    # Iterate over sorted boundaries and extract the corresponding text parts
    sorted_boundaries = sorted([(start, end, components_types[key], key) 
                                for key, bounds in components_boundaries.items() 
                                for start, end in bounds])

    for start, end, label_type, comp_id in sorted_boundaries:
        # Add the non-labeled part between the previous boundary and the current one
        if last_pos < start:
            non_labeled_part = text[last_pos:start]
            if non_labeled_part.strip():  # Avoid empty parts
                labeled_parts.append((non_labeled_part, 0))

        # Label only the first argument of a relation as '1'
        if comp_id in arg1_set:
            labeled_parts.append((text[start:end], 1))
        else:
            labeled_parts.append((text[start:end], 2))

        last_pos = end

    # Add the remaining part of the text (after the last component)
    if last_pos < len(text):
        remaining_part = text[last_pos:]
        if remaining_part.strip():
            labeled_parts.append((remaining_part, 0))

    return labeled_parts

def label_sentences_relations(text: str, components_boundaries: dict, components_types: dict, relations: list):
    """
    Labels the text into sentences by splitting it based on component boundaries and tokenizing remaining parts.

    Parameters:
        text (str): The original text to process.
        components_boundaries (dict): Mapping of component IDs to boundaries (start, end positions).
        components_types (dict): Mapping of component IDs to their types.
        relations (list): A list of abstract's relations between argumentative components.

    Returns:
        list: A list of dictionaries, where each entry contains:
            - 'sentence' (str): A sentence from the text.
            - 'label' (int): The label for the sentence (0 for None, 1 for Claim/Premise).
    """
    labeled_sentences = []

    # Split text based on boundaries and label the first argument of a relation
    labeled_parts = split_by_boundaries_relations(text, components_boundaries, components_types, relations)

    # Now process each part
    for part, label in labeled_parts:
        if label == 0:
            # For the non-labeled parts, split into sentences
            sentences = sent_tokenize(part)
            for sentence in sentences:
                labeled_sentences.append({'sentence': sentence, 'label': 0})
        elif label == 2:
            labeled_sentences.append({'sentence': part, 'label': 0})
        else:
            # For labeled parts (first argument of a relation), treat the entire part as one sentence
            labeled_sentences.append({'sentence': part, 'label': label})

    return labeled_sentences

def read_brat_dataset_relations(folder: str):
    """
    Reads BRAT dataset annotations and extracts labeled sentences.

    Parameters:
        folder (str): Path to the folder containing .ann and .txt files.

    Returns:
        list: A dataset where each entry is a dictionary with:
            - 'text' (str): The full text of the document.
            - 'filename' (str): The name of the file.
            - 'sentence' (str): The labeled sentence.
            - 'label' (int): The label for the sentence (0 or 1).
    """
    dataset, temp_sentence, merge = [], "", False
    already_in_relations = 0
    
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.ann'):
                ann_path = os.path.join(root, file)
                txt_path = ann_path.replace('.ann', '.txt')
                
                if os.path.exists(txt_path):
                    with open(ann_path, 'r', encoding='utf-8') as ann_f, open(txt_path, 'r', encoding='utf-8') as txt_f:
                        annotations = [line.strip().split('\t') for line in ann_f]
                        text = txt_f.read()

                        # Extract boundaries and types of components
                        components_boundaries = {
                            ann[0]: [(int(ann[1].split(' ')[1]), int(ann[1].split(' ')[2]))] 
                            for ann in annotations if ann[0].startswith('T')
                        }
                        components_types = {
                            ann[0]: ann[1].split(' ')[0] 
                            for ann in annotations if ann[0].startswith('T')
                        }
                        
                        # Extract relations
                        relations = []
                        for ann in annotations:
                            if ann[0].startswith('R'):
                                relation_type, arg1, arg2 = ann[1].split(' ')
                                arg1_id = arg1.split(':')[1]
                                arg2_id = arg2.split(':')[1]

                                if arg1_id in [rel['arg1_id'] for rel in relations]:
                                    already_in_relations += 1
                                    
                                relations.append({'arg1_id': arg1_id, 'arg2_id': arg2_id, 'relation_type': relation_type})
                                
                        # Label sentences based on boundaries, then split the remaining text into sentences
                        labeled_sentences = label_sentences_relations(text, components_boundaries, components_types, relations)
                        
                        for sentence in labeled_sentences:
                            if len(sentence['sentence'].strip().split()) < 3 and sentence['label'] == 0:
                                # Short sentences that are not labeled should be merged with the next
                                temp_sentence += sentence['sentence'].strip() + " "
                                merge = True
                            else:
                                if merge:
                                    temp_sentence += sentence['sentence'].strip()
                                    merge = False
                                else:
                                    temp_sentence = sentence['sentence'].strip()

                                dataset.append({
                                    'text': text,
                                    'filename': file.split('.')[0], 
                                    'sentence': temp_sentence,
                                    'label': sentence['label']  # Label only the first argument as 1
                                })

                                temp_sentence = ""
                
    # print(f"It happens {already_in_relations} times that a component is already in a relation in {folder.split('/')[len(folder.split('/'))-1]}.")  
    
    return dataset

def compute_dataset_statistics_relations(dataset: list, dataset_name: str = "dataset", label_for_component: int = 1, label_for_non_component: int = 0):
    """
    Computes and displays statistics for a labeled dataset.

    Parameters:
        dataset (list): A dataset where each element contains 'sentence', 'label', and 'filename'.
        dataset_name (str): The name of the dataset (e.g., 'train', 'test') for display (default: "dataset").
        label_for_component (int): Label representing a "component" in the dataset that is first argument in a relation(default: 1).
        label_for_non_component (int): Label representing negative class (default: 0).

    Returns:
        tuple: Contains:
            - label_counts (dict): A dictionary with counts for each label.
            - avg_sentences_per_file (float): The average number of sentences per file.
    """
    print(f'- {dataset_name.capitalize()} set:')
    max_len, max_sentences = 0, 0
    label_counts = {}
    
    # Initialize counts for components and non-components per file
    filename_to_components = defaultdict(int)
    filename_to_non_components = defaultdict(int)
    
    # Iterate over the dataset
    for data in dataset:
        # Update max sentence length
        sentence_len = len(data['sentence'].strip().split(" "))
        if sentence_len > max_len:
            max_len = sentence_len

        # Count the labels
        label = data['label']
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

        # Count components and non-components per filename
        filename = data['filename']
        if label == label_for_component:
            filename_to_components[filename] += 1
        elif label == label_for_non_component:
            filename_to_non_components[filename] += 1

    for filename in filename_to_components.keys():
        n_sentences = filename_to_components[filename] + filename_to_non_components[filename]
        if n_sentences > max_sentences:
            max_sentences = n_sentences

    # Print label counts
    for label, count in sorted(label_counts.items()):
        print(f'\tLabel {label}: {count} samples')

    # Calculate average number of sentences, components, and non-components per file
    avg_sentences_per_file = round(pd.DataFrame(dataset).groupby('filename').size().mean())
    avg_components_per_file = sum(filename_to_components.values()) / len(filename_to_components)
    avg_non_components_per_file = sum(filename_to_non_components.values()) / len(filename_to_non_components)

    # Display statistics
    print(f'\n\tThere are {len(label_counts)} different labels in the {dataset_name} set -> {sorted(list(label_counts.keys()))}')
    print(f'\tAverage number of sentences per file in {dataset_name} set: {avg_sentences_per_file}')
    print(f'\tMax sentence length: {max_len}')
    print(f'\tMax sentences in a single abstract: {max_sentences}')
    print(f'\tAverage relationships per file: {avg_components_per_file:.2f}')
    print(f'\tAverage no relationships per file: {avg_non_components_per_file:.2f}\n')

    return label_counts, avg_sentences_per_file

# NUMBER OF ARGUMENTS DATASET

def count_labels(dataset: dict, name: str):
    """
    Counts and displays the frequency of argument labels in the dataset.

    Parameters:
        dataset (dict): Dictionary where each value contains information about arguments.
        name (str): Name of the dataset for labeling the output.
    """
    label_counts = Counter(value['n'] for value in dataset.values())
    labels = sorted(label_counts.keys())
    counts = [label_counts[label] for label in labels]

    # Print summary
    print(f'Labels in {name}:')
    print('-' * 50)
    print(f'{"N.Args":<10}' + ''.join([f'{label:<10}' for label in labels]))
    print(f'{"Count":<10}' + ''.join([f'{count:<10}' for count in counts]))
    print()

def process_relations_arguments(
    filename: str,
    annotations: list,
    components_boundaries: dict,
    components_types: dict
):
    """
    Processes relations from BRAT annotations, merging them into arguments when possible.

    Parameters:
        filename (str): Name of the file being processed.
        annotations (list): List of BRAT annotation lines split into components.
        components_boundaries (dict): Mapping from component IDs to their boundary spans.
        components_types (dict): Mapping from component IDs to their types.

    Notes:
        - Arguments are formed by merging related components.
        - Components are sorted based on their boundaries.
    """
    relations = []
    
    # Parse annotations and create relations
    for ann in annotations:
        if ann[0].startswith('R'):
            rel_info = ann[1]
            _, arg1, arg2 = rel_info.split(' ')
            arg1 = arg1.split(':')[1]
            arg2 = arg2.split(':')[1]

            relations.append({
                'id': ann[0],
                'args': [arg1, arg2],
                'args_boundaries': [components_boundaries[arg1], components_boundaries[arg2]],
                'args_types': [components_types[arg1], components_types[arg2]]
            })
    
    arguments = []
    used_relations = set()  # Track processed relations

    # Create a function to continuously merge relations that have common arguments
    def merge_relations():
        merged_relations = []
        available_relations = list(range(len(relations)))

        while available_relations:
            current_idx = available_relations.pop(0)
            if current_idx in used_relations:
                continue  # Skip already processed relations

            # Start with the current relation
            args = relations[current_idx]['args'][:]
            boundaries = relations[current_idx]['args_boundaries'][:]
            types = relations[current_idx]['args_types'][:]
            merged_relations_set = {relations[current_idx]['id']}

            # Try to merge with other relations until no more can be merged
            merged = True
            while merged:
                merged = False
                for idx in list(available_relations):
                    if idx in used_relations:
                        continue  # Skip already processed relations
                    # If any argument is common, merge them
                    if set(args) & set(relations[idx]['args']):
                        for arg, boundary, type_ in zip(relations[idx]['args'], relations[idx]['args_boundaries'], relations[idx]['args_types']):
                            if arg not in args:  # Add only unique arguments
                                args.append(arg)
                                boundaries.append(boundary)
                                types.append(type_)
                        merged_relations_set.add(relations[idx]['id'])
                        available_relations.remove(idx)  # Remove the merged relation
                        merged = True

            # After merging all possible relations, mark them as used
            used_relations.update(merged_relations_set)

            # Sort arguments by their boundary start values
            sorted_info = sorted(zip(args, boundaries, types), key=lambda x: x[1][0])

            # Unzip and remove inner lists
            sorted_args, sorted_boundaries, sorted_types = zip(*sorted_info)

            count_claims = len([1 for el in sorted_types if el in ['Claim', 'MajorClaim']])
            count_premises = len([1 for el in sorted_types if el in ['Premise']])

            if count_claims and count_premises:
                arguments.append({
                    'id': f'A{len(arguments)+1}',  # Generate a new argument ID
                    'args': list(sorted_args),
                    'args_boundaries': list(sorted_boundaries),
                    'args_types': list(sorted_types)
                })

    # Start merging relations
    merge_relations()

    return relations, arguments

def filename_to_arguments_number(folder: str, threshold: int = None):
    """
    Processes all files in the specified folder to count the number of arguments.

    Parameters:
        folder (str): Path to the folder containing BRAT files.
        threshold (int, optional): Minimum frequency of arguments for files to be included.

    Notes:
        - Counts argument occurrences per file and applies a threshold if specified.
        - Displays a summary of argument counts.
    """
    dataset = {}

    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.ann'):
                ann_path = os.path.join(root, file)
                txt_path = ann_path.replace('.ann', '.txt')
                
                if os.path.exists(txt_path):
                    filename = file.split('.')[0]
                    dataset[filename] = {}
                    with open(ann_path, 'r', encoding='utf-8') as ann_f, open(txt_path, 'r', encoding='utf-8') as txt_f:
                        annotations = [line.strip().split('\t') for line in ann_f]

                        components_boundaries = {ann[0]: (int(ann[1].split(' ')[1]), int(ann[1].split(' ')[2])) 
                                                 for ann in annotations if ann[0].startswith('T')}
                        components_types = {ann[0]: ann[1].split(' ')[0] 
                                            for ann in annotations if ann[0].startswith('T')}
                       
                        relations, arguments = process_relations_arguments(filename, annotations, components_boundaries, components_types)
                        
                        dataset[filename]['filename'] = filename
                        dataset[filename]['relations'] = relations
                        dataset[filename]['arguments'] = arguments
                        dataset[filename]['n'] = len(arguments)

    # Count occurrences of each argument count
    argument_counts = Counter(value['n'] for value in dataset.values())

    # Remove entries where the occurrence of their 'n' value is below the threshold
    if threshold is not None:
        print(f"Removing files whose argument count occurs less than {threshold} times.")
        dataset = {key: value for key, value in dataset.items() if argument_counts[value['n']] >= threshold}

    count_labels(dataset, folder)

    return dataset

# AUXILIAR VISUALIZATION FUNCTIONS

def wrap_text(text: str, width: int = 120):
    """
    Wraps the given text to the specified width for better readability.

    Parameters:
        text (str): The text to wrap.
        width (int): The maximum line width (default: 120).

    Returns:
        str: The wrapped text.
    """
    return '\n'.join(textwrap.wrap(text, width=width))

def get_instances_by_filename(dataset: list, filename: str):
    """
    Filters dataset entries by filename.

    Parameters:
        dataset (list): The dataset containing data entries as dictionaries.
        filename (str): The filename to filter by.

    Returns:
        list: A list of dataset entries that match the specified filename.
    """

    """
    Selects all elements from the dataset that match the given filename.
    
    Parameters:
    - dataset: The list of data entries.
    - filename: The file name to filter by.
    
    Returns:
    - A list of elements (dicts) that have the specified filename.
    """
    return [el for el in dataset if el['filename'] == filename]

def display_file_info(dataset: list, filename: str = None, width: int = 120, show_text: bool = True, show_sentences: bool = True):
    """
    Displays the text and labeled sentences from a dataset for a given filename.

    Parameters:
        dataset (list): The dataset containing data entries.
        filename (str): The filename to display data for. If None, a random file is chosen (default: None).
        width (int): The maximum line width for wrapping sentences (default: 120).
        show_text (bool): Whether to display the full text of the file (default: True).
        show_sentences (bool): Whether to display the labeled sentences in tabular form (default: True).

    Returns:
        None
    """
    filename = filename if filename else random.choice([el['filename'] for el in dataset])    

    selected_data = get_instances_by_filename(dataset, filename)
    
    text = ''
    data = []
    for el in selected_data:
        text = el['text']
        if show_sentences:
            row = {key: field for key, field in el.items() if key not in ['text', 'filename']}
            row['label'] = 'Not a component (0)' if row['label'] == 0 else 'Component (1)' if row['label'] == 1 else ''
            row['sentence'] = wrap_text(row['sentence'], width=width)  # Wrap sentences
            data.append(row)
        else:
            break
        
    if show_text:
        print(f"File {filename} - Text:\n{text}\n")
    
    if show_sentences:
        df = pd.DataFrame(data)
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

def plot_training_history(history: dict):
    """
    Plots training and validation metrics over epochs.

    Parameters:
        history (dict): Dictionary containing metrics for training and validation across epochs.

    Notes:
        - Uses Plotly for interactive visualizations.
        - Metrics include loss, accuracy, and F1 scores.
    """
    metrics = [
        'tr-loss', 'tr-acc', 'tr-f1', 'tr-f1-w',
        'va-loss', 'va-acc', 'va-f1', 'va-f1-w',
    ]
    
    # Initialize a list for the plots
    fig = make_subplots(
        rows=2, cols=4, 
        subplot_titles=[f"Training {metric.capitalize()}" for metric in metrics[:4]] +
                       [f"Validation {metric.capitalize()}" for metric in metrics[4:]]
    )

    # Plot training metrics
    for i, metric in enumerate(metrics[:4]):  # Training metrics
        fig.add_trace(
            go.Scatter(x=history['epochs'], y=history[metric], mode='lines+markers', name=f'Train {metric.split("-")[1]}', line=dict(color='blue')),
            row=1, col=i+1
        )
        
    # Plot validation metrics
    for i, metric in enumerate(metrics[4:]):  # Validation metrics
        fig.add_trace(
            go.Scatter(x=history['epochs'], y=history[metric], mode='lines+markers', name=f'Val {metric.split("-")[1]}', line=dict(color='green')),
            row=2, col=i+1
        )

    # Set titles for x and y axes for each subplot
    for i in range(4):
        fig.update_xaxes(title_text="Epochs", row=1, col=i+1)
        fig.update_yaxes(title_text="Metric Value", row=1, col=i+1)
        
        fig.update_xaxes(title_text="Epochs", row=2, col=i+1)
        fig.update_yaxes(title_text="Metric Value", row=2, col=i+1)

    # Update layout for better presentation
    fig.update_layout(
        title="Training and Validation Metrics Over Epochs",
        height=600, width=1150,
        showlegend=False
    )

    fig.show()

def plot_training_history_per_class(history: dict, n_classes: int):
    """
    Visualizes training and validation metrics for accuracy and F1 score per class.

    Parameters:
        history (dict): Dictionary containing metrics for each class.
        n_classes (int): Number of classes in the dataset.

    Notes:
        - Creates subplots for each class and metric.
        - Provides a detailed view of model performance per class.
    """
    fig = make_subplots(
        rows=4, cols=n_classes,
        subplot_titles=[f"Train Accuracy (Class {label})" for label in range(n_classes)] +
                       [f"Val Accuracy (Class {label})" for label in range(n_classes)] +
                       [f"Train F1 Score (Class {label})" for label in range(n_classes)] +
                       [f"Val F1 Score (Class {label})" for label in range(n_classes)],
    )

    for label in range(n_classes):
        # Training Accuracy
        fig.add_trace(
            go.Scatter(x=history['epochs'], y=history[f'tr-acc-{label}'], mode='lines+markers', name=f'Train Acc (Class {label})', line=dict(color='blue')),
            row=1, col=label+1
        )
        
        # Validation Accuracy
        fig.add_trace(
            go.Scatter(x=history['epochs'], y=history[f'va-acc-{label}'], mode='lines+markers', name=f'Val Acc (Class {label})', line=dict(color='green')),
            row=2, col=label+1
        )

        # Training F1 Score
        fig.add_trace(
            go.Scatter(x=history['epochs'], y=history[f'tr-f1-{label}'], mode='lines+markers', name=f'Train F1 (Class {label})', line=dict(color='blue')),
            row=3, col=label+1
        )

        # Validation F1 Score
        fig.add_trace(
            go.Scatter(x=history['epochs'], y=history[f'va-f1-{label}'], mode='lines+markers', name=f'Val F1 (Class {label})', line=dict(color='green')),
            row=4, col=label+1
        )

        # Update axes titles
        fig.update_xaxes(title_text="Epochs", row=1, col=label+1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=label+1)
        
        fig.update_xaxes(title_text="Epochs", row=2, col=label+1)
        fig.update_yaxes(title_text="Accuracy", row=2, col=label+1)
        
        fig.update_xaxes(title_text="Epochs", row=3, col=label+1)
        fig.update_yaxes(title_text="F1 Score", row=3, col=label+1)
        
        fig.update_xaxes(title_text="Epochs", row=4, col=label+1)
        fig.update_yaxes(title_text="F1 Score", row=4, col=label+1)

    # Update layout
    fig.update_layout(
        title="Training and Validation Metrics Per Class",
        height=1200, width=1150,  # Adjust height to fit 4 rows
        showlegend=False
    )
    
    fig.show()

def index(data, indexer, inplace: bool = False, fit: bool = True, **kwargs):
    """
    Transforms textual data into numerical indices for use in machine learning models.
    Supports datasets with training, validation, and test splits, as well as single collections.
    It is a modification of QuaPy's indexing function.

    Parameters:
        data (qp.data.Dataset or qp.data.LabelledCollection): 
            Input data to be indexed. Can be a dataset with training/validation/test splits 
            or a single labelled collection.
        indexer (object): Token-to-index transformer, such as a CountVectorizer instance.
        inplace (bool): Whether to modify the input data directly (default is False).
        fit (bool): Whether to fit the indexer on the data before transforming (default is True).
        **kwargs: Additional arguments passed to the indexer.

    Notes:
        - Rare words with frequencies below a specified threshold (`min_df`) are replaced with `UNK`.
        - Handles FilenameLabelledCollection objects with support for filenames.
        - The vocabulary is stored in the data after indexing.
    """
    if isinstance(data, qp.data.Dataset):
        # Check types of training and test data instances
        qp.data.preprocessing.__check_type(data.training.instances, np.ndarray, str)
        qp.data.preprocessing.__check_type(data.test.instances, np.ndarray, str)

        # Fit or transform based on `fit` parameter
        training_index = indexer.fit_transform(data.training.instances) if fit else indexer.transform(data.training.instances)
        val_index = indexer.transform(data.val.instances)
        test_index = indexer.transform(data.test.instances)

        # Convert to numpy arrays
        training_index = np.asarray(training_index, dtype=object)
        val_index = np.asarray(val_index, dtype=object)
        test_index = np.asarray(test_index, dtype=object)

        if inplace:
            data.training = FilenameLabelledCollection(training_index, data.training.labels, data.training.filenames, data.classes_)
            data.val = FilenameLabelledCollection(val_index, data.val.labels, data.val.filenames, data.classes_)
            data.test = FilenameLabelledCollection(test_index, data.test.labels, data.test.filenames, data.classes_)
            data.vocabulary = indexer.vocabulary_
            return data
        else:
            training = FilenameLabelledCollection(training_index, data.training.labels.copy(), data.training.filenames, data.classes_)
            test = FilenameLabelledCollection(val_index, data.val.labels.copy(), data.val.filenames, data.classes_)
            test = FilenameLabelledCollection(test_index, data.test.labels.copy(), data.test.filenames, data.classes_)
            return CustomDataset(training, test, indexer.vocabulary_)
    
    elif isinstance(data, (qp.data.LabelledCollection, FilenameLabelledCollection)):
        # Check type of instances
        qp.data.preprocessing.__check_type(data.instances, np.ndarray, str)

        # Fit or transform on the single collection
        data_index = indexer.fit_transform(data.instances) if fit else indexer.transform(data.instances)
        data_index = np.asarray(data_index, dtype=object)

        if inplace:
            data.instances = data_index
            data.vocabulary = indexer.vocabulary_
            return data
        else:
            return FilenameLabelledCollection(data_index, data.labels.copy(), data.filenames, data.classes_)
    
    else:
        raise TypeError("Unsupported data type. Expected qp.data.Dataset, qp.data.LabelledCollection, or FilenameLabelledCollection.")

def objective(trial, claims_cnn_classifier, premises_cnn_classifier, relations_cnn_classifier, abs_dataset_claims: qp.data.Dataset, abs_dataset_premises: qp.data.Dataset, abs_dataset_relations: qp.data.Dataset, train_filename_to_labels: dict, val_filename_to_labels: dict, class_weights: dict, class_weights_2: dict):
    """
    Objective function for optuna, used to maximize validation on arguments predictor model.

    Parameters:
        trial: 
            Optuna's trial.
        claims_cnn_classifier:
            Claims CNN.
        premises_cnn_classifier:
            Premises CNN.
        relations_cnn_classifier:
            Relations CNN.
        abs_dataset_claims (qp.data.Dataset): 
            Input data, a dataset with training/validation/test splits.
        abs_dataset_premises (qp.data.Dataset): 
            Input data, a dataset with training/validation/test splits.
        abs_dataset_relations (qp.data.Dataset): 
            Input data, a dataset with training/validation/test splits.
        train_filename_to_labels (dict):
            Dictionary with number of arguments for each file in training set. 
        val_filename_to_labels (dict):
            Dictionary with number of arguments for each file in validation set.
        class_weights (dict):
            Class weights according to their frequency in the dataset. Used when the trials chooses to use weighted cross entropy.
        class_weights_2 (dict):
            Class weights according to their frequency in the dataset. Used when the trials chooses to use weighted random sampling.
    Returns:
        Best f1 score on validation set for the trial.
    """
    # Suggest feedforward layers
    n_ff_layers = trial.suggest_int("n_ff_layers", 2, 3)
    ap_ff_layers = [trial.suggest_categorical("ap_ff_layers0", [1024, 512, 256, 128])]
    for n in range(2, n_ff_layers+1):
        ap_ff_layers.append(ap_ff_layers[n-2]//2)

    # Suggest frozen layer percentages
    c_frozen_layers_percentage = trial.suggest_categorical('c_frozen_layers_percentage', [0, 25, 50, 100])
    p_frozen_layers_percentage = trial.suggest_categorical('p_frozen_layers_percentage', [0, 25, 50, 100])
    r_frozen_layers_percentage = trial.suggest_categorical('r_frozen_layers_percentage', [0, 25, 50, 100])
    
    # Suggest optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)  # Universal weight decay
    if optimizer_name == "Adam":
        optimizer_class = optim.Adam
        beta1 = trial.suggest_float("beta1", 0.8, 0.99)
        beta2 = trial.suggest_float("beta2", beta1, 0.999)
        optimizer_params = {"betas": (beta1, beta2), "weight_decay": weight_decay}
    elif optimizer_name == "AdamW":
        optimizer_class = optim.AdamW
        beta1 = trial.suggest_float("beta1", 0.8, 0.99)
        beta2 = trial.suggest_float("beta2", beta1, 0.999)
        optimizer_params = {"betas": (beta1, beta2), "weight_decay": weight_decay}
    else:  # SGD
        optimizer_class = optim.SGD
        optimizer_params = {"momentum": trial.suggest_float("momentum", 0.5, 0.9), "weight_decay": weight_decay}

    # Suggest learning rate
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    
    # Suggest scheduler
    scheduler_name = trial.suggest_categorical("scheduler", [None, "CosineAnnealing", "CosineAnnealingWarmRestarts"])
    if scheduler_name == "CosineAnnealing":
        scheduler_class = CosineAnnealingLR
        scheduler_params = {"T_max": trial.suggest_int("T_max", 10, 50)}
    elif scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler_class = CosineAnnealingWarmRestarts
        scheduler_params = {"T_0": trial.suggest_int("T_0", 5, 20), "T_mult": trial.suggest_int("T_mult", 1, 3)}
    else:
        scheduler_class = None
        scheduler_params = {}

    # Suggest batch size
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    
    # Weighted Random Sampler
    wrs = trial.suggest_categorical("WRS", [True, False])
    if wrs:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion=torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    # Define ArgumentsPredictorTrainer
    arguments_predictor = ArgumentsPredictorTrainerCP(
        claims_cnn_classifier,
        premises_cnn_classifier,
        relations_cnn_classifier,
        p_frozen_layers_percentage=p_frozen_layers_percentage,
        c_frozen_layers_percentage=c_frozen_layers_percentage,
        r_frozen_layers_percentage=r_frozen_layers_percentage,
        patience=12,
        qdrop_p=0,
        apdrop_p=trial.suggest_float("apdrop_p", 0, 0.5),
        ap_ff_layers=ap_ff_layers,
        batch_size=batch_size,
        qp_lr=lr,
        qc_lr=lr,
        qr_lr=lr,
        ap_lr=lr,
        criterion=criterion,
        class_weights= class_weights_2 if wrs else None,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        scheduler_class=scheduler_class,
        scheduler_params=scheduler_params,
        checkpointdir=f'../checkpoints/arguments_cp_2/optuna/02_12_24-va-f1/trial_{trial.number}/',
    )

    # Log parameters
    debug_print(
        f"Trial {trial.number}:\n"
        f"------------------------\n"
        f"Learning Rate: {lr}\n"
        f"Optimizer: {optimizer_name}\n"
        f"  - Weight Decay: {weight_decay}\n"
        f"  - Optimizer Parameters: {optimizer_params}\n"
        f"Feedforward Layers: {ap_ff_layers}\n"
        f"Batch Size: {batch_size}\n"
        f"Frozen Layers Percentage:\n"
        f"  - Claims: {c_frozen_layers_percentage}%\n"
        f"  - Premises: {p_frozen_layers_percentage}%\n"
        f"  - Relations: {r_frozen_layers_percentage}%\n"
        f"Scheduler: {scheduler_name}\n"
        f"  - Scheduler Parameters: {scheduler_params}\n"
        f"WRS: {wrs}\n"
        f"------------------------\n",
        filename=os.path.join(arguments_predictor.checkpointdir, arguments_predictor.checkpointname + '.txt')
    )

    # Train and evaluate the model
    _, best_results, _ = arguments_predictor.fit(
        abs_dataset_claims.training,
        abs_dataset_claims.val,
        abs_dataset_claims.test,
        abs_dataset_premises.training,
        abs_dataset_premises.val,
        abs_dataset_premises.test,
        abs_dataset_relations.training,
        abs_dataset_relations.val,
        abs_dataset_relations.test,
        train_filename_to_labels,
        val_filename_to_labels,
        monitor={'metric': 'va-f1', 'lower_is_better': False},
    )

    # Return the validation metric for maximization
    return best_results['va-f1']

def save_datasets(train_data: dict, val_data: dict, test_data: dict, source_dirs: list, output_dir: str = "../data/custom_datasets"):
    """
    Copies the annotations and text files for the new training, validation, and test splits into separate directories.

    Parameters:
        train_data (dict): Mapping of {filename: label} for the training set.
        val_data (dict): Mapping of {filename: label} for the validation set.
        test_data (dict): Mapping of {filename: label} for the test set.
        source_dirs (list): List of directories to locate the original files.
        output_dir (str): Path to save the datasets (default: "../data/custom_datasets").

    Notes:
        - Files are organized into "train", "val", and "test" subdirectories.
        - Looks for `.txt` and `.ann` files in the specified source directories.
        - Warns if a file cannot be located in the provided sources.
    """
    # Define subdirectories for train, validation, and test sets
    dataset_dirs = {
        "train": os.path.join(output_dir, "train"),
        "val": os.path.join(output_dir, "val"),
        "test": os.path.join(output_dir, "test")
    }

    # Create target directories
    for target_dir in dataset_dirs.values():
        os.makedirs(target_dir, exist_ok=True)

    # Helper function to locate a file in source directories
    def find_file_in_sources(filename, extensions):
        for source_dir in source_dirs:
            for ext in extensions:
                full_path = os.path.join(source_dir, f"{filename}{ext}")
                if os.path.exists(full_path):
                    return full_path
        return None

    # Helper function to copy dataset files
    def copy_dataset_files(dataset, target_dir):
        for base_filename in dataset.keys():  # Assuming `base_filename` is without extension
            for ext in [".txt", ".ann"]:
                source_file = find_file_in_sources(base_filename, [ext])
                if source_file:
                    shutil.copy(source_file, target_dir)
                else:
                    print(f"Warning: File {base_filename}{ext} not found in source directories.")

    # Save datasets to their respective directories
    for set_name, dataset in zip(["train", "val", "test"], [train_data, val_data, test_data]):
        target_dir = dataset_dirs[set_name]
        copy_dataset_files(dataset, target_dir)
        print(f"{set_name.capitalize()} dataset saved to {target_dir}")

class CustomDataset(qp.data.Dataset):
    """
    This is a "custom version" of a quapy Dataset in which we can manually set a validation set along with the training and test ones. 
    """    
    def __init__(self, training: qp.data.LabelledCollection, test: qp.data.LabelledCollection, val: qp.data.LabelledCollection, vocabulary: dict = None, name=''):
        super().__init__(training, test, vocabulary, name)
        assert set(training.classes_) == set(val.classes_), 'incompatible labels in training and val collections'
        assert set(val.classes_) == set(test.classes_), 'incompatible labels in val and test collections'
        self.val = val

class FilenameLabelledCollection(qp.data.LabelledCollection):
    """
    A class that extends `LabelledCollection` to include filenames. 

    Provides methods for sampling instances based on class prevalence or specific filenames.

    Attributes:
        instances (array-like): 
            Collection of instances, supported formats include `np.ndarray`, `list`, or `csr_matrix`.
        labels (array-like): 
            Labels corresponding to each instance, must have the same length as `instances`.
        filenames (array-like): 
            Filenames corresponding to each instance, must have the same length as `instances`.
        classes (list, optional): 
            List of classes from which labels are drawn. If not provided, classes are inferred from the labels.
    """
    def __init__(self, instances, labels, filenames=None, classes=None):
        super().__init__(instances, labels, classes=classes)
        self.filenames = np.asarray(filenames) if filenames is not None else np.array([None] * len(labels))

    def split_stratified(self, train_prop=0.6, random_state=42):
        """
        Returns two instances of :class:`FilenameLabelledCollection` split with stratification from this collection, 
        at the desired proportion.

        :param train_prop: the proportion of elements to include in the left-most returned collection (typically used
            as the training collection). The rest of the elements are included in the right-most returned collection
            (typically used as a test collection).
        :param random_state: if specified, guarantees reproducibility of the split.
        :return: two instances of :class:`FilenameLabelledCollection`, the first one with `train_prop` elements, and the
            second one with `1-train_prop` elements.
        """
        c1_docs, c2_docs, c1_labels, c2_labels, c1_filenames, c2_filenames = train_test_split(
            self.instances, self.labels, self.filenames, train_size=train_prop, stratify=self.labels, random_state=random_state
        )
        
        c1 = FilenameLabelledCollection(c1_docs, c1_labels, filenames=c1_filenames, classes=self.classes_)
        c2 = FilenameLabelledCollection(c2_docs, c2_labels, filenames=c2_filenames, classes=self.classes_)
        return c1, c2

    def split_stratified_by_filenames(self, train_prop=0.6, random_state=42):
        """
        Returns two instances of :class:`FilenameLabelledCollection` split with stratification by filenames.

        :param train_prop: The proportion of unique filenames to include in the training set.
        :param random_state: If specified, guarantees reproducibility of the split.
        :return: Two instances of :class:`FilenameLabelledCollection`, one for training and one for validation.
        """
        # Get unique filenames and their corresponding labels
        unique_filenames, filename_indices = np.unique(self.filenames, return_inverse=True)
        filename_labels = np.array([self.labels[filename_indices == i][0] for i in range(len(unique_filenames))])

        # Perform stratified split on unique filenames
        train_filenames, val_filenames = train_test_split(
            unique_filenames, train_size=train_prop, stratify=filename_labels, random_state=random_state
        )

        # Create masks for the split
        train_mask = np.isin(self.filenames, train_filenames)
        val_mask = ~train_mask

        # Create the training and validation collections
        train_collection = FilenameLabelledCollection(
            self.instances[train_mask],
            self.labels[train_mask],
            filenames=self.filenames[train_mask],
            classes=self.classes_
        )
        val_collection = FilenameLabelledCollection(
            self.instances[val_mask],
            self.labels[val_mask],
            filenames=self.filenames[val_mask],
            classes=self.classes_
        )

        return train_collection, val_collection
                    
class ScheduledNeuralClassifierTrainer(NeuralClassifierTrainer):
    """
    Extends `NeuralClassifierTrainer` to include an optional learning rate scheduler.

    Attributes:
        net (TextClassifierNet): 
            The neural network model (e.g., CNN) to be trained.
        lr_scheduler (optional): 
            A learning rate scheduler for adjusting the learning rate during training.
        optim (optional): 
            The optimizer used for training. This will be shared with the scheduler if provided.
    """
    def __init__(self, net: 'TextClassifierNet', lr_scheduler=None, optim=None, **kwargs):
        super().__init__(net, **kwargs)
        self.scheduler = lr_scheduler
        self.optim = optim  # External optimizer passed here
        
        if self.scheduler:
            self.scheduler.optimizer = self.optim

    def _train_epoch(self, data, status, pbar, epoch):
        """Train for a single epoch, applying the scheduler step at the end of each epoch if provided."""
        self.net.train()
        criterion = torch.nn.CrossEntropyLoss()
        losses, predictions, true_labels = [], [], []

        for xi, yi in data:
            self.optim.zero_grad()
            logits = self.net.forward(xi)
            loss = criterion(logits, yi)
            loss.backward()
            self.optim.step()

            # Step the scheduler per batch if it’s not ReduceLROnPlateau
            if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()

            # Update training loss and accuracy
            losses.append(loss.item())
            preds = torch.softmax(logits, dim=-1).detach().cpu().numpy().argmax(axis=-1)
            predictions.extend(preds.tolist())
            true_labels.extend(yi.detach().cpu().numpy().tolist())
            status["loss"] = np.mean(losses)
            status["acc"] = accuracy_score(true_labels, predictions)
            status["f1"] = f1_score(true_labels, predictions, average='macro')
            self.__update_progress_bar(pbar, epoch)

    def fit(self, train_instances, train_labels, val_instances, val_labels, val_split=0.3):
        """
        Fits the model with an optional learning rate scheduler.
        
        :param instances: list of lists of indexed tokens
        :param labels: array-like of shape `(n_samples, n_classes)` with the class labels
        :param val_split: proportion of training data to be used as validation set (default 0.3)
        :return: self
        """
        # Train/validation split
        # train, val = FilenameLabelledCollection(instances, labels).split_stratified(1 - val_split)
        train = FilenameLabelledCollection(train_instances, train_labels)
        val = FilenameLabelledCollection(val_instances, val_labels)
        best_val_loss, best_val_f1 = 0., 0.
        self.classes_ = train.classes_

        # Initialize data generators
        opt = self.trainer_hyperparams
        train_generator = qp.classification.neural.TorchDataset(train.instances, train.labels).asDataloader(
            opt['batch_size'], shuffle=True, pad_length=opt['padding_length'], device=opt['device'])
        valid_generator = qp.classification.neural.TorchDataset(val.instances, val.labels).asDataloader(
            opt['batch_size_test'], shuffle=False, pad_length=opt['padding_length'], device=opt['device'])

        # Initialize tracking variables
        self.status = {'tr': {'loss': -1, 'acc': -1, 'f1': -1},
                       'va': {'loss': -1, 'acc': -1, 'f1': -1},
                       'best_va': {'loss': -1, 'acc': -1, 'f1': -1}}
        self.early_stop = EarlyStop(opt['patience'], lower_is_better=False)

        # Training loop with tqdm for progress display
        with tqdm(range(1, opt['epochs'] + 1)) as pbar:
            for epoch in pbar:
                self._train_epoch(train_generator, self.status['tr'], pbar, epoch)
                self._test_epoch(valid_generator, self.status['va'], pbar, epoch)

                # Check early stopping condition
                self.early_stop(self.status['va']['f1'], epoch)
                if self.early_stop.IMPROVED:
                    self.status['best_va']['loss'] = self.status['va']['loss']
                    self.status['best_va']['f1'] = self.status['va']['f1']
                    torch.save(self.net.state_dict(), self.checkpointpath)
                elif self.early_stop.STOP:
                    print(f'Training ended by patience exhausted; loading best model parameters from {self.checkpointpath} '
                          f'from epoch {self.early_stop.best_epoch}')
                    self.net.load_state_dict(torch.load(self.checkpointpath, weights_only=True))
                    break

        # Final training pass over validation set
        print('Performing a final training pass over the validation set...')
        self._train_epoch(valid_generator, self.status['tr'], pbar, epoch=0)
        print(f'[Training complete] - Best loss on validation set: {self.status['best_va']['loss']} - Best f1 on validation set: {self.status['best_va']['f1']}')

        return self

    def __update_progress_bar(self, pbar, epoch):
        pbar.set_description(f'[{self.net.__class__.__name__}] epoch={epoch} lr={self.optim.param_groups[0]["lr"]:.5f} '
                             f'tr-loss={self.status["tr"]["loss"]:.5f} '
                             f'tr-F1={100 * self.status["tr"]["f1"]:.2f}% '
                             f'patience={self.early_stop.patience}/{self.early_stop.PATIENCE_LIMIT} '
                             f'val-loss={self.status["va"]["loss"]:.5f} '
                             f'val-F1={100 * self.status["va"]["f1"]:.2f}%')

class CustomQuaNetModule(qp.method._neural.QuaNetModule):
    """
    Extends the `QuaNet <https://dl.acm.org/doi/abs/10.1145/3269206.3269287>`_ implementation 
    in QuaPy to provide easy access to the quantification embedding produced by the last 
    layer before the fully connected ones.

    This class implements the forward pass for QuaNet. Refer to :class:`QuaNetTrainer` 
    for training procedures.

    Attributes:
        doc_embedding_size (int): 
            The dimensionality of the document embeddings.
        n_classes (int): 
            The number of classes.
        stats_size (int): 
            The number of statistics estimated by simple quantification methods.
        lstm_hidden_size (int): 
            The hidden dimensionality of the LSTM cell.
        lstm_nlayers (int): 
            The number of LSTM layers.
        ff_layers (list[int]): 
            Dimensions of the densely-connected feed-forward (FF) layers on top 
            of the quantification embedding.
        bidirectional (bool): 
            Indicates whether to use a bidirectional LSTM.
        qdrop_p (float): 
            The dropout probability.
        order_by (int): 
            The class for which the document embeddings are sorted.
    """
    def __init__(self,
                 doc_embedding_size,
                 n_classes,
                 stats_size,
                 lstm_hidden_size=64,
                 lstm_nlayers=1,
                 ff_layers=[1024, 512],
                 bidirectional=True,
                 qdrop_p=0.5,
                 order_by=0):

        super().__init__(doc_embedding_size, n_classes, stats_size, lstm_hidden_size, lstm_nlayers,
                ff_layers, bidirectional, qdrop_p, order_by)
        self.stats_size = stats_size 

    def quant_embedding(self, doc_embeddings, doc_posteriors, statistics):
        device = self.device
        
        doc_posteriors = np.array(doc_posteriors)
        doc_embeddings = torch.as_tensor(doc_embeddings, dtype=torch.float, device=device)
        doc_posteriors = torch.as_tensor(doc_posteriors, dtype=torch.float, device=device)
        statistics = torch.as_tensor(statistics, dtype=torch.float, device=device)
        
        if self.order_by is not None:
            order = torch.argsort(doc_posteriors[:, self.order_by])
            doc_embeddings = doc_embeddings[order]
            doc_posteriors = doc_posteriors[order]

        embeded_posteriors = torch.cat((doc_embeddings, doc_posteriors), dim=-1)

        # the entire set represents only one instance in quapy contexts, and so the batch_size=1
        # the shape should be (1, number-of-instances, embedding-size + n_classes)
        embeded_posteriors = embeded_posteriors.unsqueeze(0)

        self.lstm.flatten_parameters()
        _, (rnn_hidden,_) = self.lstm(embeded_posteriors, self._init_hidden())
        rnn_hidden = rnn_hidden.view(self.nlayers, self.ndirections, 1, self.hidden_size)
        quant_embedding = rnn_hidden[0].view(-1)
        quant_embedding = torch.cat((quant_embedding, statistics))

        abstracted = quant_embedding.unsqueeze(0)

        return abstracted

class ArgumentsPredictorTrainerCP(qp.method.base.BaseQuantifier):
    """
    Trainer for the ArgumentsPredictor model, combining three QuaNet models, for claims, premises and relationship quantification,
    and training a new head to predict argument count. This version is trained on the new split, hence it requries also 
    test data of the backbones as some of those instances could be part of the new training and validation sets. 
    
    Args:
        claims_classifier (Classifier): A classifier model for claims classification.
        premises_classifier (Classifier): A classifier model for premises classification.
        relations_classifier (Classifier): A classifier model for relations classification.
        batch_size (int, optional): Batch size for training. Default is 8.
        n_epochs (int, optional): Number of epochs for training. Default is 100.
        qc_lr (float, optional): Learning rate for the claims classifier. Default is 1e-3.
        qp_lr (float, optional): Learning rate for the premises classifier. Default is 1e-3.
        qr_lr (float, optional): Learning rate for the relations classifier. Default is 1e-3.
        ap_lr (float, optional): Learning rate for the arguments predictor. Default is 1e-3.
        lstm_hidden_size (int, optional): Hidden size for LSTM layers. Default is 64.
        lstm_nlayers (int, optional): Number of layers in the LSTM. Default is 1.
        ff_layers (list, optional): List of hidden layer sizes for the feedforward network. Default is [1024, 512].
        bidirectional (bool, optional): Whether to use a bidirectional LSTM. Default is True.
        qdrop_p (float, optional): Dropout probability for QuaNet. Default is 0.5.
        patience (int, optional): Number of epochs with no improvement before stopping. Default is 10.
        c_frozen_layers_percentage (int, optional): Percentage of the claims classifier layers to freeze. Default is 100.
        p_frozen_layers_percentage (int, optional): Percentage of the premises classifier layers to freeze. Default is 100.
        r_frozen_layers_percentage (int, optional): Percentage of the relations classifier layers to freeze. Default is 100.
        apdrop_p (float, optional): Dropout probability for the arguments predictor. Default is 0.5.
        ap_ff_layers (list, optional): List of hidden layer sizes for the arguments predictor. Default is [128, 64, 32].
        device (str, optional): The device to run the training on, either 'cpu' or 'cuda'. Default is 'cuda' if available.
        checkpointdir (str, optional): Directory to save checkpoints. Default is '../checkpoints/arguments/'.
        checkpointname (str, optional): Name for the checkpoint file. Default is None, which generates a timestamp-based name.
        criterion (torch.nn.Module, optional): Loss function for training. Default is CrossEntropyLoss.
        class_weights (Tensor, optional): Class weights for weighted random sampling. Default is None.
        optimizer_class (torch.optim.Optimizer, optional): Optimizer class to use. Default is Adam.
        optimizer_params (dict, optional): Parameters for the optimizer. Default is an empty dictionary.
        scheduler_class (torch.optim.lr_scheduler, optional): Scheduler class for learning rate adjustment. Default is None.
        scheduler_params (dict, optional): Parameters for the scheduler. Default is an empty dictionary.
    """

    def __init__(self, 
                claims_classifier, 
                premises_classifier, 
                relations_classifier,
                batch_size = 8,
                n_epochs=100,
                qc_lr=1e-3, 
                qp_lr=1e-3, 
                qr_lr=1e-3, 
                ap_lr=1e-3, 
                lstm_hidden_size=64,
                lstm_nlayers=1,
                ff_layers=[1024, 512],
                bidirectional=True,
                qdrop_p=0.5,
                patience=10,
                c_frozen_layers_percentage = 100, 
                p_frozen_layers_percentage = 100, 
                r_frozen_layers_percentage = 100,
                apdrop_p = 0.5,
                ap_ff_layers = [128, 64, 32],
                device=("cuda" if torch.cuda.is_available() else "cpu"), 
                checkpointdir='../checkpoints/arguments_cp_2/',
                checkpointname=None,
                criterion = torch.nn.CrossEntropyLoss(),
                class_weights=None,
                optimizer_class = torch.optim.Adam,
                optimizer_params = {},
                scheduler_class = None,  
                scheduler_params = {}):
        """
        Initializes the ArgumentsPredictorTrainerCP with the given parameters.
        """
                
        self.device = torch.device(device)
        self.class_weights = class_weights
        self.claims_classifier = claims_classifier
        self.premises_classifier = premises_classifier
        self.relations_classifier = relations_classifier
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.qp_lr = qp_lr
        self.qc_lr = qc_lr
        self.qr_lr = qr_lr
        self.ap_lr = ap_lr
        self.quanet_params = {
            'lstm_hidden_size': lstm_hidden_size,
            'lstm_nlayers': lstm_nlayers,
            'ff_layers': ff_layers,
            'bidirectional': bidirectional,
            'qdrop_p': qdrop_p
        }
        self.patience = patience
        self.p_frozen_layers_percentage = p_frozen_layers_percentage
        self.c_frozen_layers_percentage = c_frozen_layers_percentage
        self.r_frozen_layers_percentage = r_frozen_layers_percentage
        self.apdrop_p = apdrop_p
        self.ap_ff_layers = ap_ff_layers
        
        if checkpointname is None:
            timestamp = time.strftime('%d-%m-%Y_%H-%M', time.localtime()) 
            self.checkpointname = 'ArgumentsPredictor-CP-'+ timestamp
            self.checkpoint_claims_name = 'quanet_claims-' + timestamp
            self.checkpoint_premises_name = 'quanet_premises-' + timestamp
            self.checkpoint_relations_name = 'quanet_relations-'+ timestamp

        self.checkpointdir = checkpointdir
        os.makedirs(self.checkpointdir, exist_ok=True)
        self.checkpoint = os.path.join(checkpointdir, self.checkpointname+'.pth')
        self.checkpoint_claims = os.path.join(checkpointdir, self.checkpoint_claims_name+'.pth')
        self.checkpoint_premises = os.path.join(checkpointdir, self.checkpoint_premises_name+'.pth')
        self.checkpoint_relations = os.path.join(checkpointdir, self.checkpoint_relations_name+'.pth')

        self.criterion = criterion

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        self.scheduler_class = scheduler_class
        self.scheduler_params = scheduler_params

    def freeze_quanet_layers(self, model: torch.nn.Module, freeze_percentage: int):
        """
        Freezes a percentage of a QuaNet model's parameters, starting from the last layers, based on `freeze_percentage`.
        
        Args:
            model (torch.nn.Module): The QuaNet model whose layers are to be frozen.
            freeze_percentage (int): Percentage of layers to freeze, starting from the last layers.
        """
        layers = list(model.children())
        num_layers = len(layers)
        layers_to_freeze = int(num_layers * (freeze_percentage / 100.0))

        # Freeze the calculated percentage of layers, starting from the last layers
        for i, layer in enumerate(reversed(layers)):
            if i < layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

    def _get_aggregative_estims(self, posteriors, quantifiers: dict):
        """
        Aggregates predictions from different quantifiers and returns the estimations.
        
        Args:
            posteriors (ndarray): The posterior probabilities predicted by the models.
            quantifiers (dict): A dictionary of quantifiers used to aggregate the predictions.
        """
        label_predictions = np.argmax(posteriors, axis=-1)
        prevs_estim = []
        for quantifier in quantifiers.values():
            predictions = posteriors if isinstance(quantifier, qp.method.aggregative.AggregativeSoftQuantifier) else label_predictions
            prevs_estim.extend(quantifier.aggregate(predictions))

        # there is no real need for adding static estims like the TPR or FPR from training since those are constant

        return prevs_estim

    def get_filenames_dict(self, filename_to_labels: dict, collections: list[tuple]):
        """
        Processes datasets and organizes them into a dictionary keyed by filenames.
        
        Args:
            filename_to_labels (dict): Dictionary with filenames as keys and labels as values.
            collections (list of tuples): Each tuple contains a collection and its corresponding posteriors.
                                        Example: [(claims_collection, claims_posteriors), ...]
        
        Returns:
            dict: A dictionary containing organized data grouped by filenames.
        """
              
        def process_file_data(collection, posteriors, filename, instance_key, posterior_key, filenames_dict):
            """Adds data to the dictionary for the given filename if found in the collection."""
            for i, f in enumerate(collection.filenames):
                if f == filename:
                    filenames_dict[instance_key][filename].append(collection.instances[i])
                    filenames_dict[posterior_key][filename].append(posteriors[i])
        
        # Initialize the filenames dictionary
        filenames_dict = {
            'to_claims_instances': defaultdict(list),
            'to_premises_instances': defaultdict(list),
            'to_relations_instances': defaultdict(list),
            'to_claims_posteriors': defaultdict(list),
            'to_premises_posteriors': defaultdict(list),
            'to_relations_posteriors': defaultdict(list),
            'labels': {}
        }
        
        # Iterate over filenames and process data
        for filename, label in filename_to_labels.items():
            for collection, posteriors, instance_key, posterior_key in collections:
                process_file_data(collection, posteriors, filename, instance_key, posterior_key, filenames_dict)
            filenames_dict['labels'][filename] = label['n']
        
        return filenames_dict

    def fit(self, 
            claims_train_data: FilenameLabelledCollection, 
            claims_val_data: FilenameLabelledCollection,
            claims_test_data: FilenameLabelledCollection,
            premises_train_data: FilenameLabelledCollection, 
            premises_val_data: FilenameLabelledCollection, 
            premises_test_data: FilenameLabelledCollection, 
            relations_train_data: FilenameLabelledCollection, 
            relations_val_data: FilenameLabelledCollection, 
            relations_test_data: FilenameLabelledCollection, 
            train_filename_to_labels: dict,
            val_filename_to_labels: dict,
            monitor = {'metric': 'va-f1-f', 'lower_is_better': False},
            seed = 42
        ):
        """
        Trains the ArgumentsPredictorCP model on the provided training data.
        
        Args:
            claims_train_data (FilenameLabelledCollection): Training data for claims classification.
            claims_val_data (FilenameLabelledCollection): Validation data for claims classification.
            claims_test_data (FilenameLabelledCollection): Test data for claims classification.
            premises_train_data (FilenameLabelledCollection): Training data for premises classification.
            premises_val_data (FilenameLabelledCollection): Validation data for premises classification.
            premises_test_data (FilenameLabelledCollection): Test data for premises classification.
            relations_train_data (FilenameLabelledCollection): Training data for relations classification.
            relations_val_data (FilenameLabelledCollection): Validation data for relations classification.
            relations_test_data (FilenameLabelledCollection): Test data for relations classification.
            train_filename_to_labels (dict): Dictionary mapping filenames to training labels.
            val_filename_to_labels (dict): Dictionary mapping filenames to validation labels.
            monitor (dict, optional): Metric to monitor for early stopping. Default is {'metric': 'va-f1-f', 'lower_is_better': False}.
            seed (int, optional): Random seed for reproducibility. Default is 42.
        """

        def format_lr(lr):
            return "{:.2E}".format(lr)

        set_seed(seed)
        
        self._claims_classes_ = claims_train_data.classes_
        self._premises_classes_ = premises_train_data.classes_
        self._relations_classes_ = relations_train_data.classes_

        # estimate the hard and soft stats tpr and fpr of the classifier
        self.claims_tr_prev = claims_train_data.prevalence()
        self.premises_tr_prev = premises_train_data.prevalence()
        self.relations_tr_prev = relations_train_data.prevalence()

        # compute the posterior probabilities of the instances
        claims_train_posteriors = self.claims_classifier.predict_proba(claims_train_data.instances)
        claims_valid_posteriors = self.claims_classifier.predict_proba(claims_val_data.instances)
        claims_test_posteriors = self.claims_classifier.predict_proba(claims_test_data.instances)
        claims_train_data_embed = FilenameLabelledCollection(self.claims_classifier.transform(claims_train_data.instances), claims_train_data.labels, claims_train_data.filenames, self._claims_classes_)
        claims_valid_data_embed = FilenameLabelledCollection(self.claims_classifier.transform(claims_val_data.instances), claims_val_data.labels, claims_val_data.filenames, self._claims_classes_)
        claims_test_data_embed = FilenameLabelledCollection(self.claims_classifier.transform(claims_test_data.instances), claims_test_data.labels, claims_test_data.filenames, self._claims_classes_)

        premises_train_posteriors = self.premises_classifier.predict_proba(premises_train_data.instances)
        premises_valid_posteriors = self.premises_classifier.predict_proba(premises_val_data.instances)
        premises_test_posteriors = self.premises_classifier.predict_proba(premises_test_data.instances)
        premises_train_data_embed = FilenameLabelledCollection(self.premises_classifier.transform(premises_train_data.instances), premises_train_data.labels, premises_train_data.filenames, self._premises_classes_)
        premises_valid_data_embed = FilenameLabelledCollection(self.premises_classifier.transform(premises_val_data.instances), premises_val_data.labels, premises_val_data.filenames, self._premises_classes_)
        premises_test_data_embed = FilenameLabelledCollection(self.premises_classifier.transform(premises_test_data.instances), premises_test_data.labels, premises_test_data.filenames, self._premises_classes_)

        relations_train_posteriors = self.relations_classifier.predict_proba(relations_train_data.instances)
        relations_valid_posteriors = self.relations_classifier.predict_proba(relations_val_data.instances)
        relations_test_posteriors = self.relations_classifier.predict_proba(relations_test_data.instances)
        relations_train_data_embed = FilenameLabelledCollection(self.relations_classifier.transform(relations_train_data.instances), relations_train_data.labels, relations_train_data.filenames, self._relations_classes_)
        relations_valid_data_embed = FilenameLabelledCollection(self.relations_classifier.transform(relations_val_data.instances), relations_val_data.labels, relations_val_data.filenames, self._relations_classes_)
        relations_test_data_embed = FilenameLabelledCollection(self.relations_classifier.transform(relations_test_data.instances), relations_test_data.labels, relations_test_data.filenames, self._relations_classes_)

        self.claims_quantifiers = {
            'cc': qp.method.aggregative.CC(self.claims_classifier).fit(None, fit_classifier=False),
            'acc': qp.method.aggregative.ACC(self.claims_classifier).fit(None, fit_classifier=False, val_split=claims_val_data),
            'pcc': qp.method.aggregative.PCC(self.claims_classifier).fit(None, fit_classifier=False),
            'pacc': qp.method.aggregative.PACC(self.claims_classifier).fit(None, fit_classifier=False, val_split=claims_val_data),
        }

        self.premises_quantifiers = {
            'cc': qp.method.aggregative.CC(self.premises_classifier).fit(None, fit_classifier=False),
            'acc': qp.method.aggregative.ACC(self.premises_classifier).fit(None, fit_classifier=False, val_split=premises_val_data),
            'pcc': qp.method.aggregative.PCC(self.premises_classifier).fit(None, fit_classifier=False),
            'pacc': qp.method.aggregative.PACC(self.premises_classifier).fit(None, fit_classifier=False, val_split=premises_val_data),
        }

        self.relations_quantifiers = {
            'cc': qp.method.aggregative.CC(self.relations_classifier).fit(None, fit_classifier=False),
            'acc': qp.method.aggregative.ACC(self.relations_classifier).fit(None, fit_classifier=False, val_split=relations_val_data),
            'pcc': qp.method.aggregative.PCC(self.relations_classifier).fit(None, fit_classifier=False),
            'pacc': qp.method.aggregative.PACC(self.relations_classifier).fit(None, fit_classifier=False, val_split=relations_val_data),
        }

        # Istantiate and load weight for claims' QuaNet
        nQ = len(self.claims_quantifiers)
        nC = claims_train_data.n_classes
        self.claims_quanet = CustomQuaNetModule(
            doc_embedding_size=claims_train_data_embed.instances.shape[1],
            n_classes=claims_train_data.n_classes,
            stats_size=nQ*nC,
            order_by=0 if claims_train_data.binary else None,
            **self.quanet_params
        ).to(self.device)
        self.claims_quanet.load_state_dict(torch.load('../checkpoints/claims/Quanet-Claims', weights_only=True))
        self.freeze_quanet_layers(self.claims_quanet, self.c_frozen_layers_percentage)

        # Istantiate and load weight for premises' QuaNet
        nQ = len(self.premises_quantifiers)
        nC = premises_train_data.n_classes
        self.premises_quanet = CustomQuaNetModule(
            doc_embedding_size=premises_train_data_embed.instances.shape[1],
            n_classes=premises_train_data.n_classes,
            stats_size=nQ*nC,
            order_by=0 if premises_train_data.binary else None,
            **self.quanet_params
        ).to(self.device)
        self.premises_quanet.load_state_dict(torch.load('../checkpoints/premises/Quanet-Premises', weights_only=True))
        self.freeze_quanet_layers(self.premises_quanet, self.p_frozen_layers_percentage)

        # Istantiate and load weight for relations' QuaNet
        nQ = len(self.relations_quantifiers)
        nC = relations_train_data.n_classes
        self.relations_quanet = CustomQuaNetModule(
            doc_embedding_size=relations_train_data_embed.instances.shape[1],
            n_classes=relations_train_data.n_classes,
            stats_size=nQ*nC,
            order_by=0 if relations_train_data.binary else None,
            **self.quanet_params
        ).to(self.device)
        self.relations_quanet.load_state_dict(torch.load('../checkpoints/relations/Quanet-Relations', weights_only=True))
        self.freeze_quanet_layers(self.relations_quanet, self.r_frozen_layers_percentage)

        # Istantiate the new module
        n_classes = max([el['n'] for el in train_filename_to_labels.values()]) + 1
        self.arguments_predictor = ArgumentsPredictorCP(self.claims_quanet, self.premises_quanet, self.relations_quanet, n_classes, dropout_p = self.apdrop_p, ff_layers=self.ap_ff_layers)
        
        self.status = {
            'tr-loss': -1,
            'tr-acc': -1,
            'tr-f1': -1,
            'tr-f1-w': -1,
            'va-loss': -1,
            'va-acc': -1,
            'va-f1': -1,
            'va-f1-w': -1,
        }

        self.best_results = {
            'epoch': -1,
            'tr-loss': -1,
            'tr-acc': -1,
            'tr-f1': -1,
            'tr-f1-w': -1,
            'va-loss': -1,
            'va-acc': -1,
            'va-f1': -1,
            'va-f1-w': -1,
        }

        self.history = {
            'epochs': [],
            'tr-loss': [],
            'tr-acc': [],
            'tr-f1': [],
            'tr-f1-w': [],
            'va-loss': [],
            'va-acc': [],
            'va-f1': [],
            'va-f1-w': [],
        }

        for label in range(self.arguments_predictor.n_classes):
            self.status[f'tr-acc-{label}'] = -1
            self.status[f'tr-f1-{label}'] = -1
            self.status[f'va-acc-{label}'] = -1
            self.status[f'va-f1-{label}'] = -1
            
            
            self.best_results[f'tr-acc-{label}'] = -1
            self.best_results[f'tr-f1-{label}'] = -1
            self.best_results[f'va-acc-{label}'] = -1
            self.best_results[f'va-f1-{label}'] = -1
            
            self.history[f'tr-acc-{label}'] = []
            self.history[f'tr-f1-{label}'] = []
            self.history[f'va-acc-{label}'] = []
            self.history[f'va-f1-{label}'] = []

            self.optim = self.optimizer_class([
                {'params': filter(lambda p: p.requires_grad, self.claims_quanet.parameters()), 'lr': self.qc_lr},
                {'params': filter(lambda p: p.requires_grad, self.premises_quanet.parameters()), 'lr': self.qp_lr},
                {'params': filter(lambda p: p.requires_grad, self.relations_quanet.parameters()), 'lr': self.qr_lr},
                {'params': filter(lambda p: p.requires_grad, self.arguments_predictor.parameters()), 'lr': self.ap_lr}
            ], **self.optimizer_params)

        if self.scheduler_class:
            self.scheduler = self.scheduler_class(self.optim, **self.scheduler_params)

        early_stop = EarlyStop(self.patience, lower_is_better=monitor['lower_is_better'])

        collections = [
            (claims_train_data_embed, claims_train_posteriors, 'to_claims_instances', 'to_claims_posteriors'),
            (premises_train_data_embed, premises_train_posteriors, 'to_premises_instances', 'to_premises_posteriors'),
            (relations_train_data_embed, relations_train_posteriors, 'to_relations_instances', 'to_relations_posteriors'),
            (claims_valid_data_embed, claims_valid_posteriors, 'to_claims_instances', 'to_claims_posteriors'),
            (premises_valid_data_embed, premises_valid_posteriors, 'to_premises_instances', 'to_premises_posteriors'),
            (relations_valid_data_embed, relations_valid_posteriors, 'to_relations_instances', 'to_relations_posteriors'),
            (claims_test_data_embed, claims_test_posteriors, 'to_claims_instances', 'to_claims_posteriors'),
            (premises_test_data_embed, premises_test_posteriors, 'to_premises_instances', 'to_premises_posteriors'),
            (relations_test_data_embed, relations_test_posteriors, 'to_relations_instances', 'to_relations_posteriors'),
        ]

        train_filenames = self.get_filenames_dict(train_filename_to_labels, collections)
        val_filenames = self.get_filenames_dict(val_filename_to_labels, collections)
        for epoch_i in range(1, self.n_epochs):
            train_matrix = self._epoch(claims_train_data_embed, premises_train_data_embed, relations_train_data_embed, 
                        self.batch_size, train_filenames, train=True)
            val_matrix = self._epoch(claims_valid_data_embed, premises_valid_data_embed, relations_valid_data_embed, 
                        self.batch_size, val_filenames, train=False)
            
            self.history['epochs'].append(epoch_i)
            for key, value in self.status.items():
                self.history[key].append(value)

            early_stop(self.status[monitor['metric']], epoch_i)

            if early_stop.IMPROVED:
                self.best_results['epoch'] = epoch_i
                for key, value in self.status.items():
                    self.best_results[key] = value
                
                torch.save(self.claims_quanet.state_dict(), self.checkpoint_claims)
                torch.save(self.premises_quanet.state_dict(), self.checkpoint_premises)
                torch.save(self.relations_quanet.state_dict(), self.checkpoint_relations)
                torch.save(self.arguments_predictor.state_dict(), self.checkpoint)

            debug_print(
                f"[Arguments Predictor] - Epoch: {epoch_i:<3} | "
                f"QC LR: {format_lr(self.optim.param_groups[0]['lr']):<10} | "
                f"QR LR: {format_lr(self.optim.param_groups[1]['lr']):<10} | "
                f"AP LR: {format_lr(self.optim.param_groups[2]['lr']):<10}\n"
                f"\t@ Train Loss:         {self.status['tr-loss']:<10.5f}  | Val Loss:         {self.status['va-loss']:<10.5f}\n"
                f"\t@ Train Acc:          {self.status['tr-acc'] * 100:<10.2f}% | Val Acc:          {self.status['va-acc'] * 100:<10.2f}%\n"
                f"\t@ Train Macro F1:     {self.status['tr-f1']:<10.3f}  | Val Macro F1:     {self.status['va-f1']:<10.3f}\n"
                f"\t@ Train Weighted F1:  {self.status['tr-f1-w']:<10.3f}  | Val Weighted F1:  {self.status['va-f1-w']:<10.3f}\n"
                f"\n\t@ Patience: {early_stop.patience:<2}/{early_stop.PATIENCE_LIMIT:<2} "
                f"- Current best {monitor['metric']}: {self.best_results[monitor['metric']]:.5f} (epoch: {self.best_results['epoch']})",
                filename = os.path.join(self.checkpointdir, self.checkpointname+'.txt')
            )
            
            debug_print("\n\t@ Confusion matrix train:", filename = os.path.join(self.checkpointdir, self.checkpointname+'.txt')) 
            debug_print("\t" + " " * 5 + " ".join(f"{label:^7}" for label in range(4)), filename = os.path.join(self.checkpointdir, self.checkpointname+'.txt'))
            for i, row in enumerate(train_matrix):
                debug_print("\t" + f"{i:<5}" + " ".join(f"{value:^7}" for value in row), filename = os.path.join(self.checkpointdir, self.checkpointname+'.txt'))

            debug_print("\t@ Confusion matrix val:", filename = os.path.join(self.checkpointdir, self.checkpointname+'.txt')) 
            debug_print("\t" + " " * 5 + " ".join(f"{label:^7}" for label in range(4)), filename = os.path.join(self.checkpointdir, self.checkpointname+'.txt'))
            for i, row in enumerate(val_matrix):
                debug_print("\t" + f"{i:<5}" + " ".join(f"{value:^7}" for value in row), filename = os.path.join(self.checkpointdir, self.checkpointname+'.txt'))

            if early_stop.STOP:
                print(f'training ended by patience exhausted; loading best model parameters in {self.checkpoint} '
                      f'for epoch {early_stop.best_epoch} with best {monitor['metric']}: {early_stop.best_score}')
                self.claims_quanet.load_state_dict(torch.load(self.checkpoint_claims, weights_only=True))
                self.premises_quanet.load_state_dict(torch.load(self.checkpoint_premises, weights_only=True))
                self.relations_quanet.load_state_dict(torch.load(self.checkpoint_relations, weights_only=True))
                self.arguments_predictor.load_state_dict(torch.load(self.checkpoint, weights_only=True))
                break

        return self.status, self.best_results, self.history

    def _retrieve_embeddings(self, filenames: list, filenames_dict: dict, claims_data: FilenameLabelledCollection, premises_data: FilenameLabelledCollection, relations_data: FilenameLabelledCollection):
        """
        Retrieves embeddings for components and relations for the given filenames.

        Args:
            filenames (list): List of filenames to process.
            filenames_dict (dict): Dictionary containing data mappings for filenames.
            claims_data (FilenameLabelledCollection): Collection of claims with labels and filenames.
            premises_data (FilenameLabelledCollection): Collection of premises with labels and filenames.
            relations_data (FilenameLabelledCollection): Collection of relations with labels and filenames.

        Returns:
            list: A list of tuples with (claims_embedding, premises_embedding, relations_embedding, ground_truth).
        """
        embeddings = []

        for filename in filenames:
            # Retrieve instances and posteriors for the current document
            claims_sample_instances = filenames_dict['to_claims_instances'][filename]
            claims_sample_posteriors = filenames_dict['to_claims_posteriors'][filename]

            premises_sample_instances = filenames_dict['to_premises_instances'][filename]
            premises_sample_posteriors = filenames_dict['to_premises_posteriors'][filename]
            
            relations_sample_instances = filenames_dict['to_relations_instances'][filename]
            relations_sample_posteriors = filenames_dict['to_relations_posteriors'][filename]

            # Create sample data objects
            claims_sample_data = FilenameLabelledCollection(
                claims_sample_instances,
                claims_data.labels[:len(claims_sample_instances)],
                [filename] * len(claims_sample_instances)
            )
            premises_sample_data = FilenameLabelledCollection(
                premises_sample_instances,
                premises_data.labels[:len(premises_sample_instances)],
                [filename] * len(premises_sample_instances)
            )
            relations_sample_data = FilenameLabelledCollection(
                relations_sample_instances,
                relations_data.labels[:len(relations_sample_instances)],
                [filename] * len(relations_sample_instances)
            )

            # Quantifier estimates based on the sample posteriors
            claims_quant_estims = self._get_aggregative_estims(claims_sample_posteriors, self.claims_quantifiers)
            premises_quant_estims = self._get_aggregative_estims(premises_sample_posteriors, self.premises_quantifiers)
            relations_quant_estims = self._get_aggregative_estims(relations_sample_posteriors, self.relations_quantifiers)

            # Create embeddings
            claims_embedding = self.claims_quanet.quant_embedding(
                claims_sample_data.instances, claims_sample_posteriors, claims_quant_estims
            )
            premises_embedding = self.premises_quanet.quant_embedding(
                premises_sample_data.instances, premises_sample_posteriors, premises_quant_estims
            )
            relations_embedding = self.relations_quanet.quant_embedding(
                relations_sample_data.instances, relations_sample_posteriors, relations_quant_estims
            )

            # Ground truth for the document
            ground_truth = torch.as_tensor([filenames_dict['labels'][filename]], dtype=torch.long, device=self.device)

            embeddings.append((claims_embedding, premises_embedding, relations_embedding, ground_truth))

        return embeddings

    def _epoch(self, claims_data: FilenameLabelledCollection, premises_data: FilenameLabelledCollection, relations_data: FilenameLabelledCollection, batch_size: int, filenames_dict: dict, train: bool):
        """
        Runs one epoch, computes loss and evaluation metrics.

        Args:
            claims_data (FilenameLabelledCollection): Collection of claims data for the epoch.
            premises_data (FilenameLabelledCollection): Collection of premises data for the epoch.
            relations_data (FilenameLabelledCollection): Collection of relations data for the epoch.
            batch_size (int): The size of the batches for processing.
            filenames_dict (dict): Dictionary containing mappings for filenames.
            train (bool): Boolean indicating if the epoch is for training (True) or evaluation (False).

        Returns:
            np.ndarray: Confusion matrix of the evaluation results.
        """
        losses, all_gts, all_preds = [], [], []

        # Create a list of filenames
        filenames = list(filenames_dict['to_claims_instances'].keys())
        labels = [filenames_dict['labels'][filename] for filename in filenames]

        if train and self.class_weights != None:
            self.arguments_predictor.train(train)

            # Compute sample weights based on class weights
            sample_weights = np.array([self.class_weights[label] for label in labels])
            sample_probabilities = sample_weights / sample_weights.sum()

            # Sample filenames based on computed probabilities
            sampled_filenames = np.random.choice(
                filenames, size=len(filenames), replace=True, p=sample_probabilities
            )

        elif train:
            self.arguments_predictor.train(train)
            random.shuffle(filenames)
            sampled_filenames = filenames  


        else:
            self.arguments_predictor.eval()
            sampled_filenames = filenames  # No sampling during validation and evaluation

        pbar = tqdm(range(0, len(sampled_filenames) - batch_size + 1, batch_size), 
                    total=len(sampled_filenames) // batch_size, 
                    desc='Training epoch' if train else 'Validating')

        for start_idx in pbar:
            batch_filenames = sampled_filenames[start_idx:start_idx + batch_size]

            # Prepare embeddings for the current batch
            batch_embeddings = self._retrieve_embeddings(batch_filenames, filenames_dict, claims_data, premises_data, relations_data)

            claims_batch, premises_batch, relations_batch, gt_batch = zip(*batch_embeddings)
            claims_batch = torch.stack(claims_batch)
            premises_batch = torch.stack(premises_batch)
            relations_batch = torch.stack(relations_batch)
            gt_batch = torch.cat(gt_batch)

            if train:
                self.optim.zero_grad()
                pred = self.arguments_predictor.forward(claims_batch, premises_batch, relations_batch)
                loss = self.criterion(pred, gt_batch)

                loss.backward()
                self.optim.step()
            else:
                with torch.no_grad():
                    pred = self.arguments_predictor.forward(claims_batch, premises_batch, relations_batch)
                    loss = self.criterion(pred, gt_batch)

            losses.append(loss.item())
            all_gts.extend(gt_batch.detach().cpu().numpy())
            all_preds.extend(torch.argmax(pred, dim=1).detach().cpu().numpy())

        if self.scheduler_class:
            self.scheduler.step()

        loss = np.mean(losses)
        accuracy = accuracy_score(all_gts, np.round(all_preds))
        f1 = f1_score(all_gts, np.round(all_preds), average='macro')
        weighted_f1 = f1_score(all_gts, np.round(all_preds), average='weighted')

        if train:
            self.status['tr-loss'] = loss
            self.status['tr-acc'] = accuracy
            self.status['tr-f1'] = f1
            self.status['tr-f1-w'] = weighted_f1

            for label in range(self.arguments_predictor.n_classes):
                filtered_indices = [i for i, el in enumerate(all_gts) if el == label]
                filtered_gts = [all_gts[i] for i in filtered_indices]
                filtered_preds = [all_preds[i] for i in filtered_indices]
                filtered_accuracy = accuracy_score(filtered_gts, filtered_preds)
                filtered_f1 = f1_score(filtered_gts, filtered_preds, average='weighted')
                self.status[f'tr-acc-{label}'] = filtered_accuracy
                self.status[f'tr-f1-{label}'] = filtered_f1
        else:
            self.status['va-loss'] = loss
            self.status['va-acc'] = accuracy
            self.status['va-f1'] = f1
            self.status['va-f1-w'] = weighted_f1

            for label in range(self.arguments_predictor.n_classes):
                filtered_indices = [i for i, el in enumerate(all_gts) if el == label]
                filtered_gts = [all_gts[i] for i in filtered_indices]
                filtered_preds = [all_preds[i] for i in filtered_indices]
                filtered_accuracy = accuracy_score(filtered_gts, filtered_preds)
                filtered_f1 = f1_score(filtered_gts, filtered_preds, average='weighted')
                self.status[f'va-acc-{label}'] = filtered_accuracy
                self.status[f'va-f1-{label}'] = filtered_f1

        confusion_matrix = sklearn.metrics.confusion_matrix(all_gts, all_preds)

        return confusion_matrix

    def evaluate(self,
                claims_train_data: FilenameLabelledCollection,
                premises_train_data: FilenameLabelledCollection,
                relations_train_data: FilenameLabelledCollection,
                claims_val_data: FilenameLabelledCollection,
                premises_val_data: FilenameLabelledCollection,
                relations_val_data: FilenameLabelledCollection,
                claims_eval_data: FilenameLabelledCollection, 
                premises_eval_data: FilenameLabelledCollection, 
                relations_eval_data: FilenameLabelledCollection, 
                eval_filename_to_labels: dict, 
                seed: int = 42):
        """
        Evaluates the model on the given dataset and computes various metrics.

        Args:
            claims_train_data (FilenameLabelledCollection): Collection of training claims data, some of the samples could belong to the evaluation set.
            premises_train_data (FilenameLabelledCollection): Collection of training premises data, some of the samples could belong to the evaluation set.
            relations_train_data (FilenameLabelledCollection): Collection of training relations data, some of the samples could belong to the evaluation set.
            claims_val_data (FilenameLabelledCollection): Collection of validation claims data, some of the samples could belong to the evaluation set.
            premises_val_data (FilenameLabelledCollection): Collection of validation premises data, some of the samples could belong to the evaluation set.
            relations_val_data (FilenameLabelledCollection): Collection of validation relations data, some of the samples could belong to the evaluation set.
            claims_eval_data (FilenameLabelledCollection): Collection of claims data for evaluation.
            premises_eval_data (FilenameLabelledCollection): Collection of premises data for evaluation.
            relations_eval_data (FilenameLabelledCollection): Collection of relations data for evaluation.
            eval_filename_to_labels (dict): Dictionary mapping filenames to their respective labels.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            dict: Dictionary containing evaluation metrics such as accuracy, F1 score, and confusion matrix.
        """
        set_seed(seed)
        self.arguments_predictor.eval()

        claims_train_posteriors = self.claims_classifier.predict_proba(claims_train_data.instances)
        claims_valid_posteriors = self.claims_classifier.predict_proba(claims_val_data.instances)
        claims_eval_posteriors = self.claims_classifier.predict_proba(claims_eval_data.instances)
        claims_train_data_embed = FilenameLabelledCollection(self.claims_classifier.transform(claims_train_data.instances), claims_train_data.labels, claims_train_data.filenames, self._claims_classes_)
        claims_valid_data_embed = FilenameLabelledCollection(self.claims_classifier.transform(claims_val_data.instances), claims_val_data.labels, claims_val_data.filenames, self._claims_classes_)
        claims_eval_data_embed = FilenameLabelledCollection(self.claims_classifier.transform(claims_eval_data.instances), claims_eval_data.labels, claims_eval_data.filenames, self._claims_classes_)

        premises_train_posteriors = self.premises_classifier.predict_proba(premises_train_data.instances)
        premises_valid_posteriors = self.premises_classifier.predict_proba(premises_val_data.instances)
        premises_eval_posteriors = self.premises_classifier.predict_proba(premises_eval_data.instances)
        premises_train_data_embed = FilenameLabelledCollection(self.premises_classifier.transform(premises_train_data.instances), premises_train_data.labels, premises_train_data.filenames, self._premises_classes_)
        premises_valid_data_embed = FilenameLabelledCollection(self.premises_classifier.transform(premises_val_data.instances), premises_val_data.labels, premises_val_data.filenames, self._premises_classes_)
        premises_eval_data_embed = FilenameLabelledCollection(self.premises_classifier.transform(premises_eval_data.instances), premises_eval_data.labels, premises_eval_data.filenames, self._premises_classes_)

        relations_train_posteriors = self.relations_classifier.predict_proba(relations_train_data.instances)
        relations_valid_posteriors = self.relations_classifier.predict_proba(relations_val_data.instances)
        relations_eval_posteriors = self.relations_classifier.predict_proba(relations_eval_data.instances)
        relations_train_data_embed = FilenameLabelledCollection(self.relations_classifier.transform(relations_train_data.instances), relations_train_data.labels, relations_train_data.filenames, self._relations_classes_)
        relations_valid_data_embed = FilenameLabelledCollection(self.relations_classifier.transform(relations_val_data.instances), relations_val_data.labels, relations_val_data.filenames, self._relations_classes_)
        relations_eval_data_embed = FilenameLabelledCollection(self.relations_classifier.transform(relations_eval_data.instances), relations_eval_data.labels, relations_eval_data.filenames, self._relations_classes_)

        collections = [
            (claims_train_data_embed, claims_train_posteriors, 'to_claims_instances', 'to_claims_posteriors'),
            (premises_train_data_embed, premises_train_posteriors, 'to_premises_instances', 'to_premises_posteriors'),
            (relations_train_data_embed, relations_train_posteriors, 'to_relations_instances', 'to_relations_posteriors'),
            (claims_valid_data_embed, claims_valid_posteriors, 'to_claims_instances', 'to_claims_posteriors'),
            (premises_valid_data_embed, premises_valid_posteriors, 'to_premises_instances', 'to_premises_posteriors'),
            (relations_valid_data_embed, relations_valid_posteriors, 'to_relations_instances', 'to_relations_posteriors'),
            (claims_eval_data_embed, claims_eval_posteriors, 'to_claims_instances', 'to_claims_posteriors'),
            (premises_eval_data_embed, premises_eval_posteriors, 'to_premises_instances', 'to_premises_posteriors'),
            (relations_eval_data_embed, relations_eval_posteriors, 'to_relations_instances', 'to_relations_posteriors'),
        ]

        eval_filenames = self.get_filenames_dict(
            filename_to_labels=eval_filename_to_labels,
            collections=collections
        )    

        # Perform evaluation using _epoch (train=False ensures no updates)
        eval_matrix = self._epoch(
            claims_eval_data_embed, premises_eval_data_embed, relations_eval_data_embed, 
            self.batch_size, eval_filenames, train=False
        )
        
        print(
            f"[Arguments Predictor] Test-set"
            f"\t@ Loss:         {self.status['va-loss']:<10.5f}\n"
            f"\t@ Acc:          {self.status['va-acc'] * 100:<10.2f}%\n"
            f"\t@ Macro F1:     {self.status['va-f1']:<10.3f}\n"
            f"\t@ Weighted F1:  {self.status['va-f1-w']:<10.3f}\n"
        )

        print("\n\t@ Confusion matrix:") 
        print("\t" + " " * 5 + " ".join(f"{label:^7}" for label in range(4)))
        for i, row in enumerate(eval_matrix):
            print("\t" + f"{i:<5}" + " ".join(f"{value:^7}" for value in row))

        # Extract and return evaluation metrics
        metrics = {
            'Accuracy': self.status['va-acc'],
            'F1 Macro': self.status['va-f1'],
            'F1 Weighted': self.status['va-f1-w'],
            'Confusion Matrix': eval_matrix
        }

        return metrics
    
class ArgumentsPredictorCP(nn.Module):
    """
    Combines two QuaNet models for component and relationship quantification to predict the number of arguments.
    Directly uses feed-forward layers to process the combined embedding.
    """
    def __init__(self, claims_quanet, premises_quanet, relations_quanet, n_classes, dropout_p=0.5, ff_layers=[128, 64, 32]):
        super(ArgumentsPredictorCP, self).__init__()

        # Calculate combined embedding size
        claims_stats_size = claims_quanet.stats_size
        claims_output_size = claims_quanet.hidden_size * claims_quanet.ndirections
        
        premises_stats_size = premises_quanet.stats_size
        premises_output_size = premises_quanet.hidden_size * premises_quanet.ndirections
        
        relations_stats_size = relations_quanet.stats_size
        relations_output_size = relations_quanet.hidden_size * relations_quanet.ndirections
        
        self.combined_embedding_size = (claims_stats_size + claims_output_size) + (premises_stats_size + premises_output_size) + (relations_stats_size + relations_output_size)

        self.n_classes = n_classes

        # Fully connected layers
        self.dropout = nn.Dropout(dropout_p)
        self.ff_layers = nn.ModuleList()

        prev_size = self.combined_embedding_size
        for layer_size in ff_layers:
            self.ff_layers.append(nn.Linear(prev_size, layer_size))
            prev_size = layer_size
        
        # Output layer for the number of arguments prediction
        self.output_layer = nn.Linear(prev_size, self.n_classes)

    def forward(self, claims_embedding, premises_embedding, relations_embedding):
        """
        Forward pass for the ArgumentsPredictor.
        
        :param components_embedding: Tensor of shape [batch_size, components_embedding_dim].
        :param relations_embedding: Tensor of shape [batch_size, relations_embedding_dim].
        :return: Tensor of shape [batch_size, n_classes].
        """
        # Concatenate and pass through the fully connected layers
        combined_embedding = torch.cat((claims_embedding, premises_embedding, relations_embedding), dim=-1).view(claims_embedding.size(0), -1)
        x = combined_embedding

        # TEST        
        # average_embedding = (components_embedding + relations_embedding) / 2
        # x = average_embedding.view(components_embedding.size(0), -1)
        # TEST        
        
        for layer in self.ff_layers:
            x = self.dropout(nn.functional.relu(layer(x)))
        
        argument_count = self.output_layer(x)
        
        # return argument_count  
        return argument_count.squeeze()  # Remove unnecessary dimensions
