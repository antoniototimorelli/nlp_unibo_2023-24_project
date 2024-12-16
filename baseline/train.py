# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification and sequence tagging"""


import argparse
import glob
import json
import logging
import os
import random
import csv

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)
from utils.data_processors import output_modes
from utils.data_processors import processors
from utils.models import BertForSequenceTagging
from utils.metrics import compute_metrics
from utils.tokenizer import ExtendedBertTokenizer
from utils.filename_to_arguments import *
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            RobertaConfig,
            DistilBertConfig,
            AlbertConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "bert-seqtag": (BertConfig, BertForSequenceTagging, ExtendedBertTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix=""):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        decoded_texts = []
        prediction_details = []  
        processor = processors[args.task_name]()

        output_file = os.path.join(eval_output_dir, prefix, f"eval_results_{os.path.basename(os.path.normpath(args.data_dir))}.txt")
        if os.path.isfile(output_file):
            os.remove(output_file)
            logger.info("  Deleted existing evaluation file!")

        for idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )

                outputs = model(**inputs)

                if args.output_mode == "sequencetagging":
                    tmp_eval_loss, emissions, path = outputs[:3]
                    logits_tensor = path.detach().cpu().numpy()
                    labels_tensor = inputs["labels"].detach().cpu().numpy()

                    def extract_segments(indices, labels):
                        """Extract contiguous segments from a list of indices."""
                        segments = []
                        start = indices[0]
                        for i in range(1, len(indices)):
                            if indices[i] != indices[i - 1] + 1 or (labels[indices[i]] == 1 and labels[indices[i - 1]] == 2) or (labels[indices[i]] == 3 and labels[indices[i - 1]] == 4):  # Non-contiguous or different components
                                segments.append((start, indices[i - 1]))
                                start = indices[i]

                        segments.append((start, indices[-1]))  # Add the final segment
                        return segments

                    for input_ids, logits, labels in zip(inputs["input_ids"], logits_tensor, labels_tensor):
                        # Process Claims
                        claim_indices = np.where((labels == 1) | (labels == 2))[0]
                        if len(claim_indices) > 0:
                            claim_segments = extract_segments(claim_indices, labels)
                            for start, end in claim_segments:
                                segment_logits = logits[start:end + 1]
                                segment_labels = labels[start:end + 1]
                                overlap = np.sum(segment_logits == segment_labels) / len(segment_labels)
                                if overlap >= 0.75:  # Check if overlap is at least 75%
                                    claim_tokens = tokenizer.convert_ids_to_tokens(input_ids[start:end + 1].tolist())
                                    claim_tokens = [t for t in claim_tokens if t not in ["[CLS]", "[SEP]", "[PAD]"]]
                                    claim_text = tokenizer.convert_tokens_to_string(claim_tokens)
                                    decoded_texts.append((claim_text, "Claim"))

                        # Process Premises
                        premise_indices = np.where((labels == 3) | (labels == 4))[0]
                        if len(premise_indices) > 0:
                            premise_segments = extract_segments(premise_indices, labels)
                            for start, end in premise_segments:
                                segment_logits = logits[start:end + 1]
                                segment_labels = labels[start:end + 1]
                                overlap = np.sum(segment_logits == segment_labels) / len(segment_labels)
                                if overlap >= 0.75:  # Check if overlap is at least 75%
                                    premise_tokens = tokenizer.convert_ids_to_tokens(input_ids[start:end + 1].tolist())
                                    premise_tokens = [t for t in premise_tokens if t not in ["[CLS]", "[SEP]", "[PAD]"]]
                                    premise_text = tokenizer.convert_tokens_to_string(premise_tokens)
                                    decoded_texts.append((premise_text, "Premise"))

                    # Continue with your logic for writing to file, etc.
                    tokenizer.batch_to_conll(inputs["input_ids"], logits_tensor, labels_tensor, processor, output_file)


                    tokenizer.batch_to_conll(inputs["input_ids"], logits_tensor, labels_tensor, processor, output_file)
                    logits = logits_tensor.flatten()
                    labels = labels_tensor.flatten()

                elif args.task_name == "relclass":
                    tmp_eval_loss, logits = outputs[:2]
                    logits = logits.detach().cpu().numpy()
                    labels = inputs["labels"].detach().cpu().numpy()

                    for input_ids, pred, label in zip(inputs["input_ids"].tolist(), logits, labels):
                        tokens = tokenizer.convert_ids_to_tokens(input_ids)
                        sep_index = tokens.index("[SEP]") if "[SEP]" in tokens else len(tokens)
                        sentence1_tokens = [t for t in tokens[:sep_index] if t not in ["[CLS]", "[SEP]", "[PAD]"]]
                        sentence2_tokens = [t for t in tokens[sep_index + 1:] if t not in ["[CLS]", "[SEP]", "[PAD]"]]

                        sentence1 = tokenizer.convert_tokens_to_string(sentence1_tokens)
                        sentence2 = tokenizer.convert_tokens_to_string(sentence2_tokens)
                        prediction_details.append((tokens, sentence1, sentence2, np.argmax(pred, axis=0), label))

                else:
                    tmp_eval_loss, logits = outputs[:2]
                    logits = logits.detach().cpu().numpy()
                    labels = inputs["labels"].detach().cpu().numpy()

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits
                out_label_ids = labels
            else:
                preds = np.append(preds, logits, axis=0)
                out_label_ids = np.append(out_label_ids, labels, axis=0)

        eval_loss = eval_loss / nb_eval_steps

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        if args.task_name == "relclass":
            # Read the .tsv file that contains components correctly predicted (overlap > 75%) by the sequence tagging model
            ov75_dict = {}
            ov75_file = os.path.join(args.data_dir, "seqtag_results.tsv")
            if os.path.isfile(ov75_file):
                with open(ov75_file, "r", encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter="\t")  # Read the file as a TSV
                    next(reader)  # Skip the header
                    for row in reader:
                        if row[1] not in ov75_dict:
                            ov75_dict[row[1]] = {"premises": [], "claims": [], "relations": []}
                        
                        if row[2] == "Premise":
                            ov75_dict[row[1]]["premises"].append((f"P{len(ov75_dict[row[1]]['premises'])+1}", row[0]))
                        elif row[2] == "Claim":
                            ov75_dict[row[1]]["claims"].append((f"C{len(ov75_dict[row[1]]['claims'])+1}", row[0]))

            # Filter predictions and update ov75_dict with relations
            filtered_preds, filtered_out_label_ids = [], []
            for tokens, sentence1, sentence2, pred, label in prediction_details:
                overlap1, overlap2 = False, False
                arg1_name, arg2_name = None, None
                matching_filename = None

                # Check overlap for both sentences with premises and claims
                for filename, data in ov75_dict.items():
                    # Check overlap for sentence1
                    for premise in data["premises"]:
                        if sentence1 in premise[1]:
                            overlap1 = True
                            arg1_name = premise[0]
                            break
                    if not overlap1:
                        for claim in data["claims"]:
                            if sentence1 in claim[1]:
                                overlap1 = True
                                arg1_name = claim[0]
                                break
                    
                    if overlap1:
                        # Check overlap for sentence2
                        for premise in data["premises"]:
                            if sentence2 in premise[1]:
                                overlap2 = True
                                arg2_name = premise[0]
                                break
                        if not overlap2:
                            for claim in data["claims"]:
                                if sentence2 in claim[1]:
                                    overlap2 = True
                                    arg2_name = claim[0]
                                    break
                    if overlap1 and overlap2:
                        matching_filename = filename
                        break  # No need to check further once both overlaps are confirmed

                if overlap1 and overlap2 and matching_filename:
                    # Add relation to ov75_dict if correctly predicted
                    if pred and pred == label:
                        ov75_dict[matching_filename]["relations"].append(
                            {
                                "relation_id": f"R{len(ov75_dict[matching_filename]['relations'])+1}",
                                "arg1": arg1_name,
                                "arg2": arg2_name,
                            }
                        )

                    # Add to filtered lists
                    filtered_preds.append(pred)
                    filtered_out_label_ids.append(label)

            # Now we will reconstruct the arguments
            for filename, data in ov75_dict.items():
                if ov75_dict[filename]["relations"]:
                    # Initialize union-find structure for argument reconstruction
                    node_to_parent = {}
                    
                    def find(node):
                        # Find the root parent of a node
                        if node_to_parent[node] != node:
                            node_to_parent[node] = find(node_to_parent[node])  # Path compression
                        return node_to_parent[node]

                    def union(node1, node2):
                        # Union two nodes into the same set
                        root1 = find(node1)
                        root2 = find(node2)
                        if root1 != root2:
                            node_to_parent[root2] = root1

                    # Add all premises and claims as separate nodes
                    for premise in data["premises"]:
                        node_to_parent[premise[0]] = premise[0]
                    for claim in data["claims"]:
                        node_to_parent[claim[0]] = claim[0]

                    # Merge nodes based on relations
                    for relation in data["relations"]:
                        union(relation["arg1"], relation["arg2"])

                    # Group nodes by their root parent to form arguments
                    arguments = {}
                    for node in node_to_parent:
                        root = find(node)
                        if root not in arguments:
                            arguments[root] = []
                        arguments[root].append(node)

                    # Validate arguments: keep only those with at least one premise and one claim
                    valid_arguments = []
                    for nodes in arguments.values():
                        has_premise = any(node.startswith("P") for node in nodes)
                        has_claim = any(node.startswith("C") for node in nodes)
                        if has_premise and has_claim:
                            valid_arguments.append(nodes)

                    # Store valid arguments
                    data["arguments"] = valid_arguments
                    data["n"] = len(valid_arguments)
                else:
                    data["arguments"] = []
                    data["n"] = 0


            # Convert to numpy arrays for metrics computation
            filtered_preds = np.array(filtered_preds)
            filtered_out_label_ids = np.array(filtered_out_label_ids)

            # Compute metrics for relation classification task
            result_unfiltered = compute_metrics(eval_task, preds, out_label_ids)
            result_filtered = compute_metrics(eval_task, filtered_preds, filtered_out_label_ids, ov75=True)

            # Debugging: Print the updated ov75_dict
            aq_ground_truth_values = []
            aq_predicted_values = []
            # aq_labels = filename_to_arguments_number(f'../data/test/{os.path.basename(os.path.normpath(args.data_dir))}')
            aq_labels = filename_to_arguments_number(f'../data/custom_datasets/test')
            for filename, data in ov75_dict.items():
                aq_predicted_values.append(data["n"])
                aq_ground_truth_values.append(aq_labels[filename]["n"])
            
            aq_accuracy = accuracy_score(aq_ground_truth_values, aq_predicted_values)
            aq_f1 = f1_score(aq_ground_truth_values, aq_predicted_values, average="macro")  
            aq_classification_report = classification_report(aq_ground_truth_values, aq_predicted_values)
            aq_results = {"arguments_quantifier_accuracy": aq_accuracy, "arguments_quantifier_f1_score": aq_f1, "aq_classification_report": aq_classification_report}

            result = {**result_unfiltered, **result_filtered, **aq_results}

            # DEBUG
            # print(set(aq_predicted_values))
            # print(set(aq_ground_truth_values))
            # for filename, data in ov75_dict.items():
            #     print(f"File: {filename}")
            #     print(f"Premises: {data['premises']}")
            #     print(f"Claims: {data['claims']}")
            #     print(f"Relations: {data['relations']}")
            #     print(f"Arguments: {data['arguments']}")
            #     print(f"Predicted arguments: {data['n']}")
            #     print(f"Ground truth arguments: {aq_labels[filename]['n']}")
            #     print()
            #     print()

        else:
            result = compute_metrics(eval_task, preds, out_label_ids)

        results.update(result)

        with open(output_file, "w", encoding='utf-8') as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        if args.task_name == 'seqtag':
            with open(os.path.join(args.data_dir, "seqtag_decoded_texts.txt"), "w", encoding='utf-8') as f:
                for text, c_type in decoded_texts:
                    f.write(c_type + "\n" + text + "\n\n")

    return results

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate and args.do_train else processor.get_test_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )

        if args.output_mode == "sequencetagging":
            forSequenceTagging = True
        else:
            forSequenceTagging = False

        features = processor.convert_examples_to_features(examples, max_seq_length=args.max_seq_length, tokenizer=tokenizer, logger=logger, forSequenceTagging=forSequenceTagging, min_seq_length=5)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    dataset = processor.features_to_dataset(features)

    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()

    # TEST
    # test = processor.get_test_examples(args.data_dir)
    # print(test[0].guid)
    # print(test[0].text_a)
    # print(test[0].text_b)
    # print(test[0].labels)
    # TEST

    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()