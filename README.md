# Estimating Argumentative Class Prevalence in Textual Data for Automatic Analysis of Scientific Literature
In this repository you can find the code used for all the experiments conducted in the related report.

## Table of Contents
1. [Project Description](#project-description)
2. [Folder Structure](#folder-structure)

## Project Description
Peer review is a cornerstone of research but is resource-intensive. Recent studies have suggested that a research paper's relevance correlates with its argumentative content. However, classification-based approaches to evaluate argumentative content have proven suboptimal. This project addresses these limitations by focusing on quantification, directly estimating class prevalence rather than classifying individual items.

We employ datasets derived from abstracts of randomized controlled trials (RCTs) in the medical domain. By leveraging QuaNet and other state-of-the-art techniques, this work aims to estimate argumentative structures effectively, contributing to more efficient and scalable analysis of scientific literature.

## Folder Structure

```
|-- baseline/		# Baseline Implementation
|   |-- data/
|   |-- preprocessing/
|   |-- runs/
|   |-- utils/
|-- data/		# Datasets
|   |-- custom_datasets
|   |-- dev/
|   |-- test/
|   |-- train/             
|-- experiments/	# Source codes and Jupyter notebooks for each experiment
|   |-- exports/
|   |-- images/         
|-- papers/		# Useful Papers used as background        
|-- results/            # Results for each experiment conducted, divided in subfolders
|   |-- baseline/
|   |-- experiment_1_components/
|   |-- experiment_1_a_claims/
|   |-- experiment_1_b_premises/
|   |-- experiment_2_relations/
|   |-- experiment_3_arguments_quantifier/
|   |-- experiment_3_a_arguments_quantifier_cp/
|-- report.pdf		# Project report
```