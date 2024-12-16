import os, random, nltk, textwrap, optuna, torch
from tabulate import tabulate
import pandas as pd

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm
from copy import copy
from itertools import pairwise
from sklearn.model_selection import train_test_split

import quapy as qp
from quapy.data import Dataset
from quapy.method.meta import QuaNet
from quapy.classification.neural import NeuralClassifierTrainer, CNNnet
from quapy.error import ae, rae, mse, mae, mrae, mkld
import quapy.functional as F
from quapy.util import EarlyStop

from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

def set_seed(seed: int):
    """
    Sets the seed for reproducibility in random, numpy, and PyTorch libraries.

    Parameters:
        seed (int): The seed value to ensure reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def split_by_boundaries(text: str, components_boundaries: dict, components_types: dict, relations: list):
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

def label_sentences(text: str, components_boundaries: dict, components_types: dict, relations: list):
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
    labeled_parts = split_by_boundaries(text, components_boundaries, components_types, relations)

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

def read_brat_dataset(folder: str):
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
                        labeled_sentences = label_sentences(text, components_boundaries, components_types, relations)
                        
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

    print(f'It happens {already_in_relations} times that a component is already in a relation in {folder.split('/')[len(folder.split('/'))-1]}.')  
    
    return dataset

def compute_dataset_statistics(dataset: list, dataset_name: str = "dataset", label_for_component: int = 1, label_for_non_component: int = 0):
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
    max_len = 0
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
    print(f'\tAverage relationships per file: {avg_components_per_file:.2f}')
    print(f'\tAverage no relationships per file: {avg_non_components_per_file:.2f}\n')

    return label_counts, avg_sentences_per_file

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

def objective(trial, abs_dataset: qp.data.Dataset):
    """
    Objective function for optuna, used to maximize validation on relations' CNN.

    Parameters:
        trial: 
            Optuna's trial.
        abs_dataset (qp.data.Dataset): 
            Input data to be indexed. Can be a dataset with training/validation/test splits 
            or a single labelled collection.
    Returns:
        Best f1 score on validation set for the trial.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the hyperparameters to tune
    embedding_size = trial.suggest_int("embedding_size", 100, 200)
    hidden_size = trial.suggest_int("hidden_size", 256, 300)
    
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    
    scheduler_name = trial.suggest_categorical("scheduler", ["CosineAnnealingLR", "CosineAnnealingWarmRestarts"])
    if scheduler_name == "CosineAnnealingLR":
        T_max = trial.suggest_int("T_max", 2, 12)
        scheduler_params = {"T_max": T_max}
    elif scheduler_name == "CosineAnnealingWarmRestarts":
        T_0 = trial.suggest_int("T_0", 1, 12)
        T_mult = trial.suggest_int("T_mult", 1, 5)
        scheduler_params = {"T_0": T_0, "T_mult": T_mult}

    print(f"Starting trial {trial.number} with parameters:")
    print(f"    Embedding size: {embedding_size} - Hidden size: {hidden_size}")
    print(f"    Optimizer: {optimizer_name} (lr: {lr}) - Scheduler: {scheduler_name} (params: {scheduler_params})")
    
    cnn_module = CNNnet(
        abs_dataset.vocabulary_size,
        abs_dataset.training.n_classes,
        embedding_size=embedding_size,
        hidden_size=hidden_size
    )
    
    optimizer = Adam(cnn_module.parameters(), lr=lr)
    
    if scheduler_name == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, **scheduler_params)
    elif scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, **scheduler_params)
    
    cnn_classifier = ScheduledNeuralClassifierTrainer(
        cnn_module,
        lr_scheduler=scheduler,
        optim=optimizer,
        device=device,
        checkpointpath='../checkpoints/relations/classifier_net.dat',
        padding_length=107,
        patience=5
    )
    
    cnn_classifier.fit(*abs_dataset.training.Xy, *abs_dataset.val.Xy)

    # Monitoring the best obtained f1 on validation set
    return cnn_classifier.status['best_va']['f1']

def evaluate(collection: object, quantifier: object, n):
    """
    Evaluates the estimated distributions of the model both by document (filename) and using standard sampling.

    Parameters:
        collection (object): The collection containing the samples, such as a LabelledCollection.
        quantifier (object): The method used for quantification (e.g., a model or algorithm for quantification).
        n (int or list): The maximum number of files to group together for evaluation. Can be a single integer 
                         or a list of integers representing batch sizes.

    Returns:
        dict: Averaged error measures for the estimated distributions (both by document and standard).
    """
    set_seed(42)
 
    filename_to_instances = defaultdict(list)
    filename_to_labels = defaultdict(list)

    for i in range(len(collection.instances)):
        filename = collection.filenames[i]
        filename_to_instances[filename].append(collection.instances[i])
        filename_to_labels[filename].append(collection.labels[i])

    # Evaluate by standard sampling technique 
    true_dist_all = collection.prevalence()
    estim_dist_all = quantifier.quantify(collection.instances)
    
    std_ae = qp.error.ae(true_dist_all, estim_dist_all)
    std_rae = qp.error.rae(true_dist_all, estim_dist_all)
    std_mse = qp.error.mse(true_dist_all, estim_dist_all)
    std_mae = qp.error.mae(true_dist_all, estim_dist_all)
    std_mrae = qp.error.mrae(true_dist_all, estim_dist_all)
    std_mkld = qp.error.mkld(true_dist_all, estim_dist_all)

    results = {
        'Std_AE': std_ae,
        'Std_RAE': std_rae,
        'Std_MSE': std_mse,
        'Std_MAE': std_mae,
        'Std_MRAE': std_mrae,
        'Std_MKLD': std_mkld,
    }
    
    n_values = [n]  if isinstance(n, int) else n
    
    header = ['Error Metric', 'Standard'] + [f'ByDoc (n={i})' for i in n_values]
    table_data = {
        'AE': [std_ae],
        'RAE': [std_rae],
        'MSE': [std_mse],
        'MAE': [std_mae],
        'MRAE': [std_mrae],
        'MKLD': [std_mkld]
    }
        
    # Evaluate by filename, custom sampling method, in batches of size batch_size
    # Iterate through each batch size from 1 to n
    for batch_size in n_values:
        total_ae, total_rae, total_mse = 0.0, 0.0, 0.0
        total_mae, total_mrae, total_mkld = 0.0, 0.0, 0.0
        
        filenames = list(set(collection.filenames))
        n_batches = (len(filenames) + batch_size - 1) // batch_size 

        for i in range(0, len(filenames), batch_size):
            batch_filenames = filenames[i:i + batch_size]
            
            # Gather instances and labels for the current batch
            batch_instances = []
            batch_labels = []
            for filename in batch_filenames:
                batch_instances.extend(filename_to_instances[filename])
                batch_labels.extend(filename_to_labels[filename])

            # True distribution for the current batch
            true_dist = np.bincount(batch_labels, minlength=2) / len(batch_labels)
            
            # Estimated distribution from quantifier
            estim_dist = quantifier.quantify(batch_instances)
            
            total_ae += qp.error.ae(true_dist, estim_dist)
            total_rae += qp.error.rae(true_dist, estim_dist)
            total_mse += qp.error.mse(true_dist, estim_dist)
            total_mae += qp.error.mae(true_dist, estim_dist)
            total_mrae += qp.error.mrae(true_dist, estim_dist)
            total_mkld += qp.error.mkld(true_dist, estim_dist)

        # Average errors for the current batch size
        avg_ae = total_ae / n_batches if n_batches > 0 else 0
        avg_rae = total_rae / n_batches if n_batches > 0 else 0
        avg_mse = total_mse / n_batches if n_batches > 0 else 0
        avg_mae = total_mae / n_batches if n_batches > 0 else 0
        avg_mrae = total_mrae / n_batches if n_batches > 0 else 0
        avg_mkld = total_mkld / n_batches if n_batches > 0 else 0

        # Append average results to the corresponding metric
        table_data['AE'].append(avg_ae)
        table_data['RAE'].append(avg_rae)
        table_data['MSE'].append(avg_mse)
        table_data['MAE'].append(avg_mae)
        table_data['MRAE'].append(avg_mrae)
        table_data['MKLD'].append(avg_mkld)

        # Update results dictionary
        results.update({
            f'ByDoc_AE_{batch_size}': avg_ae,
            f'ByDoc_RAE_{batch_size}': avg_rae,
            f'ByDoc_MSE_{batch_size}': avg_mse,
            f'ByDoc_MAE_{batch_size}': avg_mae,
            f'ByDoc_MRAE_{batch_size}': avg_mrae,
            f'ByDoc_MKLD_{batch_size}': avg_mkld
        })

    print("\t".join(header))
    print("\t".join(["-" * len(h) for h in header]))

    for metric in table_data:
        row = [metric] + table_data[metric]
        print("\t".join(f"{value:<15.4f}" if isinstance(value, float) else f"{value:<15}" for value in row))

    return results

def infer(dataset: list[dict], indexer: object, comp_quantifier: object = None, comp_classifier: object = None, 
          rel_quantifier: object = None, rel_classifier: object = None, filename: str = None, 
          show_text: bool = True, show_sentences: bool = False, use_tokenizer: bool = True):
    """
    Performs inference by displaying analysis results based on components or relations in a dataset.

    Parameters:
        dataset (list): A list of data entries, each containing at least a 'sentence' and 'label'.
        indexer (object): The indexer used to transform text instances into a numeric format.
        comp_quantifier (optional, object): A quantifier for component analysis, defaults to None.
        comp_classifier (optional, object): A classifier for component analysis, defaults to None.
        rel_quantifier (optional, object): A quantifier for relation analysis, defaults to None.
        rel_classifier (optional, object): A classifier for relation analysis, defaults to None.
        filename (optional, str): The specific filename to analyze; if None, a random filename is selected.
        show_text (bool): Whether to display text information for the given file, defaults to True.
        show_sentences (bool): Whether to display sentence-level information for the given file, defaults to False.
        use_tokenizer (bool): Whether to tokenize the text, defaults to True.

    Returns:
        None
    """
    # Choose a random filename if one isn't provided
    if filename is None:
        filename = random.choice([el['filename'] for el in dataset])

    display_file_info(dataset, filename=filename, width=120, show_text=show_text, show_sentences=show_sentences)

    if comp_quantifier or comp_classifier:
        print("\n" + "*"*50)
        print('Component Analysis')
        process_entity('components', dataset, filename, indexer, comp_quantifier, comp_classifier, use_tokenizer)

    if rel_quantifier or rel_classifier:
        print("\n" + "*"*50)
        print('Relation Analysis')
        process_entity('relations', dataset, filename, indexer, rel_quantifier, rel_classifier, use_tokenizer)

def process_entity(entity_type: str, dataset: list[dict], filename: str, indexer: object, 
                   quantifier: object, classifier: object, use_tokenizer: bool):
    """
    Processes a specific entity (either components or relations) from the dataset and performs quantification 
    and classification tasks.

    Parameters:
        entity_type (str): The type of entity to process, either 'components' or 'relations'.
        dataset (list): A list of data entries, each containing at least a 'sentence' and 'label'.
        filename (str): The filename identifying the document to be processed.
        indexer (object): The indexer used to transform instances into a numeric format.
        quantifier (optional, object): The quantifier used for estimating the distribution of the entity, defaults to None.
        classifier (optional, object): The classifier used for predicting entity labels, defaults to None.
        use_tokenizer (bool): Whether to use a tokenizer for sentence processing, defaults to True.

    Returns:
        None
    """
    # Get instances and labels based on the entity type
    instances, labels = zip(*[(el['sentence'], el['label']) for el in get_instances_by_filename(dataset, filename)])

    # Create a collection by transforming instances using indexer
    index = np.asarray(indexer.transform(instances), dtype=object)
    collection = FilenameLabelledCollection(index, labels, [filename] * len(instances))

    # Quantification
    if quantifier:
        print('\tQuantification:')
        true_distribution = np.bincount(collection.labels, minlength=2) / len(collection.labels)
        estimated_distribution = quantifier.quantify(collection.instances)
        print(f"\t\t# True distribution {entity_type}: [Class 0 = {true_distribution[0]:.4f}, Class 1 = {true_distribution[1]:.4f}]")
        print(f"\t\t# Estimated distribution {entity_type}: [Class 0 = {estimated_distribution[0]:.4f}, Class 1 = {estimated_distribution[1]:.4f}]")

        if use_tokenizer:
            text_data = sent_tokenize(get_instances_by_filename(dataset, filename)[0]['text'])
            tokenized_instances = [s1 + s2 for (s1, s2) in combinations(text_data, 2)] if entity_type == 'relations' else text_data
            dummy_labels = [0] * len(tokenized_instances)
            index_tokenized = np.asarray(indexer.transform(tokenized_instances), dtype=object)
            collection_tokenized = FilenameLabelledCollection(index_tokenized, dummy_labels, [filename] * len(tokenized_instances))
            estimated_tokenized_distribution = quantifier.quantify(collection_tokenized.instances)
            print(f"\t# Estimated distribution on tokenized {entity_type}: [Class 0 = {estimated_tokenized_distribution[0]:.4f}, Class 1 = {estimated_tokenized_distribution[1]:.4f}]")
        
        # Calculate and print performance metrics
        metrics = {
            'AE': ae(true_distribution, estimated_distribution),
            'RAE': rae(true_distribution, estimated_distribution),
            'MSE': mse(true_distribution, estimated_distribution),
            'MAE': mae(true_distribution, estimated_distribution),
            'MRAE': mrae(true_distribution, estimated_distribution),
            'MKLD': mkld(true_distribution, estimated_distribution)
        }

        print()
        for error, value in metrics.items():
            print(f"\t# {error}: {value:.4f}")

    # Classification
    if classifier:
        print('\tClassification:')
        predictions = classifier.predict(collection.instances)
        print(f"\t\t# Ground truth {entity_type}: {labels}")
        print(f"\t\t# Predicted {entity_type} labels: {predictions}")

class CustomDataset(qp.data.Dataset):
    """
    This is a "custom version" of a quapy Dataset in which we can manually set a validation set along with the training and test ones. 
    """
    def __init__(self, training: qp.data.LabelledCollection, test: qp.data.LabelledCollection, val: qp.data.LabelledCollection, vocabulary: dict = None, name=''):
        super().__init__(training, test, vocabulary, name)
        assert set(training.classes_) == set(val.classes_), 'incompatible labels in training and val collections'
        assert set(val.classes_) == set(test.classes_), 'incompatible labels in val and test collections'
        self.val = val

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
                    self.net.load_state_dict(torch.load(self.checkpointpath))
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
    
class QuaNetTrainerABS(qp.method._neural.QuaNetTrainer):
    """
    Trainer for a QuaNet model where the training routine is executed over files rather than random samples drawn from all the sentences in the dataset.
    """
    def fit(self, data: FilenameLabelledCollection, fit_classifier=True):
        """
        Trains QuaNet.

        :param data: the training data on which to train QuaNet. If `fit_classifier=True`, the data will be split in
            40/40/20 for training the classifier, training QuaNet, and validating QuaNet, respectively. If
            `fit_classifier=False`, the data will be split in 66/34 for training QuaNet and validating it, respectively.
        :param fit_classifier: if True, trains the classifier on a split containing 40% of the data
        :return: self
        """
        self._classes_ = data.classes_
        os.makedirs(self.checkpointdir, exist_ok=True)

        if fit_classifier:
            classifier_data, unused_data = data.split_stratified(0.4)
            train_data, valid_data = unused_data.split_stratified(0.66)  # 0.66 split of 60% makes 40% and 20%
            self.classifier.fit(*classifier_data.Xy)
        else:
            classifier_data = None
            train_data, valid_data = data.split_stratified(0.66)

        # estimate the hard and soft stats tpr and fpr of the classifier
        self.tr_prev = data.prevalence()

        # compute the posterior probabilities of the instances
        valid_posteriors = self.classifier.predict_proba(valid_data.instances)
        train_posteriors = self.classifier.predict_proba(train_data.instances)

        # turn instances' original representations into embeddings
        valid_data_embed = FilenameLabelledCollection(self.classifier.transform(valid_data.instances), valid_data.labels, valid_data.filenames, self._classes_)
        train_data_embed = FilenameLabelledCollection(self.classifier.transform(train_data.instances), train_data.labels, valid_data.filenames, self._classes_)

        self.quantifiers = {
            'cc': qp.method.aggregative.CC(self.classifier).fit(None, fit_classifier=False),
            'acc': qp.method.aggregative.ACC(self.classifier).fit(None, fit_classifier=False, val_split=valid_data),
            'pcc': qp.method.aggregative.PCC(self.classifier).fit(None, fit_classifier=False),
            'pacc': qp.method.aggregative.PACC(self.classifier).fit(None, fit_classifier=False, val_split=valid_data),
        }
        if classifier_data is not None:
            self.quantifiers['emq'] = qp.method.aggregative.EMQ(self.classifier).fit(classifier_data, fit_classifier=False)

        self.status = {
            'tr-loss': -1,
            'va-loss': -1,
            'tr-mae': -1,
            'va-mae': -1,
        }

        nQ = len(self.quantifiers)
        nC = data.n_classes
        self.quanet = qp.method._neural.QuaNetModule(
            doc_embedding_size=train_data_embed.instances.shape[1],
            n_classes=data.n_classes,
            stats_size=nQ*nC,
            order_by=0 if data.binary else None,
            **self.quanet_params
        ).to(self.device)
        print(self.quanet)

        self.optim = torch.optim.Adam(self.quanet.parameters(), lr=self.lr)
        early_stop = EarlyStop(self.patience, lower_is_better=True)

        checkpoint = self.checkpoint

        for epoch_i in range(1, self.n_epochs):
            self._epoch(train_data_embed, train_posteriors, self.tr_iter, epoch_i, early_stop, train=True)
            self._epoch(valid_data_embed, valid_posteriors, self.va_iter, epoch_i, early_stop, train=False)

            early_stop(self.status['va-loss'], epoch_i)
            if early_stop.IMPROVED:
                torch.save(self.quanet.state_dict(), checkpoint)
            elif early_stop.STOP:
                print(f'training ended by patience exhausted; loading best model parameters in {checkpoint} '
                      f'for epoch {early_stop.best_epoch}')
                self.quanet.load_state_dict(torch.load(checkpoint))
                break

        return self

    def _epoch(self, data: FilenameLabelledCollection, posteriors, iterations, epoch, early_stop, train):
        mse_loss = MSELoss()

        self.quanet.train(mode=train)
        losses = []
        mae_errors = []

        # Group the data by document (filename)
        filename_to_instances = defaultdict(list)
        filename_to_posteriors = defaultdict(list)

        # Fill the dictionaries with instances and posteriors grouped by document
        for i in range(len(data.instances)):
            filename = data.filenames[i]
            filename_to_instances[filename].append(data.instances[i])
            filename_to_posteriors[filename].append(posteriors[i])

        # Create a list of filenames
        filenames = list(filename_to_instances.keys())

        # Progress bar for tracking iterations
        pbar = tqdm(filenames, total=len(filenames))

        for it, filename in enumerate(pbar):
            
            # Retrieve instances and posteriors for the current document
            sample_instances = filename_to_instances[filename]
            sample_posteriors = filename_to_posteriors[filename]

            # Create a sample data object with instances from the current document
            sample_data = FilenameLabelledCollection(sample_instances, 
                                                    data.labels[:len(sample_instances)],  # Use corresponding labels
                                                    [filename] * len(sample_instances))

            # Quantifier estimates based on the sample posteriors
            quant_estims = self._get_aggregative_estims(sample_posteriors)
            ptrue = torch.as_tensor([sample_data.prevalence()], dtype=torch.float, device=self.device)

            if train:
                self.optim.zero_grad()
                phat = self.quanet.forward(sample_data.instances, sample_posteriors, quant_estims)
                loss = mse_loss(phat, ptrue)
                mae = qp.method._neural.mae_loss(phat, ptrue)
                loss.backward()
                self.optim.step()
            else:
                with torch.no_grad():
                    phat = self.quanet.forward(sample_data.instances, sample_posteriors, quant_estims)
                    loss = mse_loss(phat, ptrue)
                    mae = qp.method._neural.mae_loss(phat, ptrue)

            losses.append(loss.item())
            mae_errors.append(mae.item())

            # Compute mean losses and MAE
            mse = np.mean(losses)
            mae = np.mean(mae_errors)
            if train:
                self.status['tr-loss'] = mse
                self.status['tr-mae'] = mae
            else:
                self.status['va-loss'] = mse
                self.status['va-mae'] = mae

            # Update progress bar description
            if train:
                pbar.set_description(f'[QuaNet] '
                                    f'epoch={epoch} [it={it}/{iterations}]\t'
                                    f'tr-mseloss={self.status["tr-loss"]:.5f} tr-maeloss={self.status["tr-mae"]:.5f}\t'
                                    f'val-mseloss={self.status["va-loss"]:.5f} val-maeloss={self.status["va-mae"]:.5f} '
                                    f'patience={early_stop.patience}/{early_stop.PATIENCE_LIMIT}')