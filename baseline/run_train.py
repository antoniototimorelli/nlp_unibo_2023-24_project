import os
import subprocess

# SELECT TASK
# 1) SEQUENCE TAGGING
TASK_NAME = "seqtag"
MODELTYPE = "bert-seqtag"

# 2) RELATION CLASSIFICATION
#TASK_NAME="relclass"
#MODELTYPE="bert"

# 3) MULTIPLE CHOICE (requires that train_multiplechoice.py is executed instead of train.py, see below)
#TASK_NAME="multichoice"
#MODELTYPE="bert-multichoice"

# PATH TO TRAINING DATA
# DATA_DIR = "data/neoplasm"
DATA_DIR = "data/custom_datasets/train"
MAXSEQLENGTH = 128
OUTPUTDIR = f"output_new_split/{TASK_NAME}+{MAXSEQLENGTH}/"

# SELECT MODEL FOR FINE-TUNING
#MODEL="bert-base-uncased"
#MODEL="monologg/biobert_v1.1_pubmed"
#MODEL="monologg/scibert_scivocab_uncased"
MODEL="allenai/scibert_scivocab_uncased"
#MODEL="roberta-base"

# COMMAND TO RUN THE TRAINING SCRIPT
command = [
    "python", "train.py",
    "--model_type", MODELTYPE,
    "--model_name_or_path", MODEL,
    "--output_dir", OUTPUTDIR,
    "--task_name", TASK_NAME,
    "--no_cuda",
    "--do_train",
    "--do_eval",
    "--do_lower_case",
    "--data_dir", DATA_DIR,
    "--max_seq_length", str(MAXSEQLENGTH),
    "--overwrite_output_dir",
    "--per_gpu_train_batch_size", "8",
    "--learning_rate", "2e-5",
    "--num_train_epochs", "3.0",
    "--save_steps", "1000",
    "--overwrite_cache"  # Required for multiple choice
]

# SETTING ENVIRONMENT VARIABLES
os.environ["TASK_NAME"] = TASK_NAME
os.environ["MODELTYPE"] = MODELTYPE
os.environ["DATA_DIR"] = DATA_DIR
os.environ["OUTPUTDIR"] = OUTPUTDIR
os.environ["MODEL"] = MODEL

# RUNNING THE COMMAND
try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error while running the training script: {e}")
