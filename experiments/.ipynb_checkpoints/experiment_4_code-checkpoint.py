import os, random, nltk, textwrap
from tabulate import tabulate
import pandas as pd

import numpy as np
import torch
from torch.nn import MSELoss
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

nltk.download('punkt_tab')

def split_by_boundaries(text, components_boundaries, components_types):
    """
    Split the text into parts based on the component boundaries and label them accordingly.
    
    This returns a list of tuples: (text_part, label), where label is:
        - 0 -> None
        - 1 -> Claim
        - 2 -> Premise
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
        labeled_parts.append((text[start:end], 1 if label_type == 'Claim' or label_type == 'MajorClaim' or label_type == 'Premise' else 0))
        
        last_pos = end

    # Add the remaining part of the text (after the last component)
    if last_pos < len(text):
        remaining_part = text[last_pos:]
        if remaining_part.strip():
            labeled_parts.append((remaining_part, 0))

    return labeled_parts

def label_sentences(text, components_boundaries, components_types):
    """
    First split the text according to the component boundaries, label premises and claims,
    and then split the remaining parts into sentences and label them as None.
    """
    labeled_sentences = []
    
    # Split text based on boundaries and label Claims/Premises
    labeled_parts = split_by_boundaries(text, components_boundaries, components_types)
    
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

def read_brat_dataset(folder):
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
                        labeled_sentences = label_sentences(text, components_boundaries, components_types)
                        
                        for sentence in labeled_sentences:
                            if len(sentence['sentence'].strip().split()) < 3 and sentence['label'] == 0:
                                # print(sentence['sentence'])z
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

def compute_dataset_statistics(dataset, dataset_name="dataset", label_for_component=1, label_for_non_component=0):
    """
    Computes statistics for a given dataset, including label counts, average number of sentences per file, 
    max sentence length, and average number of components and non-components per file.

    Parameters:
        dataset (list): The dataset where each element contains 'sentence', 'label', and 'filename'.
        dataset_name (str): The name of the dataset (e.g., 'train' or 'test') for display purposes.
        label_for_component (int): The label indicating a "component" in the dataset.
        label_for_non_component (int): The label indicating a "non-component" in the dataset.

    Returns:
        None
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
    print(f'\tAverage components per file: {avg_components_per_file:.2f}')
    print(f'\tAverage non-components per file: {avg_non_components_per_file:.2f}\n')

    return label_counts, avg_sentences_per_file

def wrap_text(text, width=120):
    return '\n'.join(textwrap.wrap(text, width=width))

def get_instances_by_filename(dataset, filename):
    """
    Selects all elements from the dataset that match the given filename.
    
    Parameters:
    - dataset: The list of data entries.
    - filename: The file name to filter by.
    
    Returns:
    - A list of elements (dicts) that have the specified filename.
    """
    return [el for el in dataset if el['filename'] == filename]

def display_file_info(dataset, filename=None, width=120, show_text=True, show_sentences=True):
    """
    Displays the text and labeled sentences from a dataset for a given filename.
    
    Parameters:
    - dataset: The list of data entries.
    - filename: The file name to filter by. If None, a random file is chosen.
    - width: The width for text wrapping (default is 50).
    
    Returns:
    - Prints the text and a table of labeled sentences for the selected file.
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

def index(dataset: Dataset, indexer, inplace=False, **kwargs):
    """
    Indexes the tokens of a textual :class:`quapy.data.base.Dataset` of string documents.
    To index a document means to replace each different token by a unique numerical index.
    Rare words (i.e., words occurring less than `min_df` times) are replaced by a special token `UNK`

    :param dataset: a :class:`quapy.data.base.Dataset` object where the instances of training and test documents
        are lists of str
    :param min_df: minimum number of occurrences below which the term is replaced by a `UNK` index
    :param inplace: whether or not to apply the transformation inplace (True), or to a new copy (False, default)
    :param kwargs: the rest of parameters of the transformation (as for sklearn's
        `CountVectorizer <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>_`)
    :return: a new :class:`quapy.data.base.Dataset` (if inplace=False) or a reference to the current
        :class:`quapy.data.base.Dataset` (inplace=True) consisting of lists of integer values representing indices.
    """
    qp.data.preprocessing.__check_type(dataset.training.instances, np.ndarray, str)
    qp.data.preprocessing.__check_type(dataset.test.instances, np.ndarray, str)

    training_index = indexer.fit_transform(dataset.training.instances)
    test_index = indexer.transform(dataset.test.instances)

    training_index = np.asarray(training_index, dtype=object)
    test_index = np.asarray(test_index, dtype=object)

    if inplace:
        dataset.training = FilenameLabelledCollection(training_index, dataset.training.labels, dataset.training.filenames, dataset.classes_)
        dataset.test = FilenameLabelledCollection(test_index, dataset.test.labels, dataset.test.filenames, dataset.classes_)
        dataset.vocabulary = indexer.vocabulary_
        return dataset
    else:
        training = FilenameLabelledCollection(training_index, dataset.training.labels.copy(), dataset.training.filenames, dataset.classes_)
        test = FilenameLabelledCollection(test_index, dataset.test.labels.copy(), dataset.test.filenames, dataset.classes_)
        return Dataset(training, test, indexer.vocabulary_)
    
def evaluate(collection, quantifier, n):
    """
    Evaluates the estimated distributions of the model both by document (filename) and using standard sampling.

    Parameters:
        collection (object): The collection containing the samples.
        quantifier (object): The method used to evaluate.
        n (int or list): The maximum number of files to group together for evaluation. Can be a single integer or a list of integers.

    Returns:
        dict: Averaged error measures for the estimated distributions (both by document and standard).
    """
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
            true_dist = np.bincount(batch_labels, minlength=len(collection.classes_)) / len(batch_labels)
            
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

def infer(dataset, indexer, comp_quantifier=None, comp_classifier=None, rel_quantifier=None, filename=None, show_text=True, show_sentences=False, use_tokenizer=True):    
    # Choose a random filename if one isn't provided
    if filename is None:
        filename = random.choice([el['filename'] for el in dataset])

    display_file_info(dataset, filename=filename, width=120, show_text=show_text, show_sentences=show_sentences)

    if comp_quantifier:
        instances, labels = zip(*[(el['sentence'], el['label']) for el in get_instances_by_filename(dataset, filename)])

        collection = FilenameLabelledCollection(list(instances), 
                                                list(labels), 
                                                [filename] * len(list(instances)))

        index = np.asarray(indexer.transform(collection.instances), dtype=object)
        collection = FilenameLabelledCollection(index, collection.labels, collection.classes_)

        true_pm_distribution = np.bincount(collection.labels, minlength=len(collection.classes_)) / len(collection.labels)
        estim_pm_distribution = comp_quantifier.quantify(collection.instances)

        print(f'\t# Labels: {labels}')

        if comp_classifier:
            preds = comp_classifier.predict(collection.instances)
            print(f'\t# Predicted labels: {preds}')
        
        print(f'\n\t# True distribution: [Non-components = {true_pm_distribution[0]}, Components = {true_pm_distribution[1]}]')
        print(f'\t# Estimated distribution: [Non-components = {estim_pm_distribution[0]}, Components = {estim_pm_distribution[1]}]\n')

        # If 'use_tokenizer' is True, run the quantifier on tokenized sentences without ground truth labels
        if use_tokenizer:
            text_data = get_instances_by_filename(dataset, filename)[0]['text']
            tokenized_sentences = sent_tokenize(text_data)
            
            # Use dummy labels (all zeros) since ground truth labels are unavailable
            dummy_labels = [0] * len(tokenized_sentences)
    
            collection = FilenameLabelledCollection(tokenized_sentences, dummy_labels, [filename] * len(tokenized_sentences))
    
            index = np.asarray(indexer.transform(collection.instances), dtype=object)
            collection = FilenameLabelledCollection(index, collection.labels, collection.classes_)
    
            estim_pm_distribution_2 = comp_quantifier.quantify(collection.instances)
            
            print(f'\t# Estimated distribution on tokenized sentences: [None = {estim_pm_distribution_2[0]}, Components = {estim_pm_distribution_2[1]}]\n')
            
        pm_result = {
            'AE': ae(true_pm_distribution,estim_pm_distribution),
            'RAE': rae(true_pm_distribution,estim_pm_distribution),
            'MSE': mse(true_pm_distribution,estim_pm_distribution),
            'MAE': mae(true_pm_distribution,estim_pm_distribution),
            'MRAE': mrae(true_pm_distribution,estim_pm_distribution),
            'MKLD': mkld(true_pm_distribution,estim_pm_distribution)
        }

        for error, value in pm_result.items():
            print(f'\t# {error}: {value:.4f}')

    if rel_quantifier:
        pass

class FilenameLabelledCollection(qp.data.LabelledCollection):
    """
    A FilenameLabelledCollection is an extension of LabelledCollection that includes filenames.
    It provides methods to sample instances based on class prevalence or from specific filenames.

    :param instances: array-like (np.ndarray, list, or csr_matrix are supported)
    :param labels: array-like with the same length as instances
    :param filenames: array-like with the same length as instances; the filenames corresponding to each instance
    :param classes: optional, list of classes from which labels are taken. If not specified, the classes are inferred
        from the labels.
    """

    def __init__(self, instances, labels, filenames=None, classes=None):
        super().__init__(instances, labels, classes=classes)
        self.filenames = np.asarray(filenames) if filenames is not None else np.array([None] * len(labels))


    def split_stratified(self, train_prop=0.6, random_state=None):
        """
        Returns two instances of :class:`LabelledCollection` split with stratification from this collection, at desired
        proportion.

        :param train_prop: the proportion of elements to include in the left-most returned collection (typically used
            as the training collection). The rest of elements are included in the right-most returned collection
            (typically used as a test collection).
        :param random_state: if specified, guarantees reproducibility of the split.
        :return: two instances of :class:`LabelledCollection`, the first one with `train_prop` elements, and the
            second one with `1-train_prop` elements
        """
        tr_docs, te_docs, tr_labels, te_labels = train_test_split(
            self.instances, self.labels, train_size=train_prop, stratify=self.labels, random_state=random_state
        )
        training = FilenameLabelledCollection(tr_docs, tr_labels, self.filenames, classes=self.classes_)
        test = FilenameLabelledCollection(te_docs, te_labels, self.filenames, classes=self.classes_)
        return training, test

class QuaNetTrainerABS(qp.method._neural.QuaNetTrainer):
    
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