import os
import subprocess

# SELECT TASK
# 1) SEQUENCE TAGGING
# TASK_NAME = "seqtag"
# MODELTYPE = "bert-seqtag"

# 2) RELATION CLASSIFICATION
# TASK_NAME="relclass"
# MODELTYPE="bert"

# 3) MULTIPLE CHOICE (requires that train_multiplechoice.py is executed instead of train.py, see below)
#TASK_NAME="multichoice"
#MODELTYPE="bert-multichoice"

# PATH TO TEST DATA
# DATA_DIR = "data/neoplasm_test/" 
# DATA_DIR = "data/glaucoma_test/"
# DATA_DIR = "data/mixed_test/"

# Define task names, model types, and other constants
task_model_combinations = [
    # ("seqtag", "bert-seqtag"),
    ("relclass", "bert")
]
# DATA_DIRS = ["data/neoplasm_test/"]
# DATA_DIRS = ["data/neoplasm_test/", "data/glaucoma_test/", "data/mixed_test/"]  
DATA_DIRS = ["data/custom_datasets/test"]  
MAXSEQLENGTH = 128

for TASK_NAME, MODELTYPE in task_model_combinations:
    MODEL = f"output_new_split/{TASK_NAME}+{MAXSEQLENGTH}/"
    OUTPUTDIR = MODEL

    for DATA_DIR in DATA_DIRS:
        # Define the command to run the evaluation script
        command = [
            "python", "train.py",
            "--model_type", MODELTYPE,
            "--model_name_or_path", MODEL,
            "--output_dir", OUTPUTDIR,
            "--task_name", TASK_NAME,
            "--no_cuda",
            "--do_eval",
            "--do_lower_case",
            "--data_dir", DATA_DIR,
            "--max_seq_length", str(MAXSEQLENGTH),
            "--overwrite_output_dir",
            "--overwrite_cache",
            "--per_gpu_train_batch_size", "8",
            "--learning_rate", "2e-5",
            "--num_train_epochs", "1.0",
            "--save_steps", "1000"
        ]

        # Setting environment variables
        os.environ["TASK_NAME"] = TASK_NAME
        os.environ["MODELTYPE"] = MODELTYPE
        os.environ["DATA_DIR"] = DATA_DIR
        os.environ["OUTPUTDIR"] = OUTPUTDIR
        os.environ["MODEL"] = MODEL

        # Run the command and handle potential errors
        try:
            print(f"Running evaluation for TASK_NAME={TASK_NAME}, MODELTYPE={MODELTYPE}, DATA_DIR={DATA_DIR}")
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error while running the evaluation script for {TASK_NAME}, {MODELTYPE}: {e}")
