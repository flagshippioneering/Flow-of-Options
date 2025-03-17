"""
A script for running an agent-based pipeline to generate and execute Python code for machine learning tasks
using Flow-of-Options under the hood.

This script uses a structured pipeline to process tasks defined in a JSON file, generate Python scripts 
according to provided templates, and evaluate their execution. The pipeline leverages configurations 
and hyperparameters set in external modules and script-level constants.

Functions:
    get_full_admet_group(json_file, task_group):
        Reads a JSON file containing groups of tasks and retrieves the tasks for a specified group.
        
    driver(
        parent_dir,
        model,
        num_options,
        num_walks_per_iter,
        num_iters,
        num_filter_steps,
        parallel_walks,
        beam_widths,
        use_cbr,
        db_path,
        do_pruning,
        use_retrieval,
        save_case,
        replace_cases
    ):
        Orchestrates the task pipeline execution, setting up directories, preparing prompts,
        and executing the automated pipeline using specified models and configurations.

Main Execution:
    The script sets random seeds for reproducibility across random operations using values from 
    a configuration file. It reads task-specific settings from a JSON file and constructs a task-specific 
    prompt incorporating an example template. The prompt is processed by the pipeline to generate 
    executable code, and metrics are logged and returned for analysis. The results, including breakdown 
    of costs and the best performance metric achieved, are printed and logged.

Use Cases:
    - Automatically generate and evaluate Python scripts for specific machine learning tasks.
    - Utilize a customizable pipeline for reproducible experimentation across various task configurations.
    - Log and monitor task execution details and metrics for further review and analysis.
    
Dependencies:
    - os, json, random, torch, numpy: Standard libraries for file operations, data manipulation, and reproducibility.
    - agent_with_options.run_pipeline: Core pipeline for generating Python scripts for tasks using Flow-of-Options.
    - config as cfg: External configuration file providing task-specific settings and parameters.
"""


import os
import json
from agent_with_options import run_pipeline
import config as cfg
import random
import torch
import numpy as np


# For Reproducibility
if cfg.SEED is not None:
    random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)


def get_full_admet_group(json_file, task_group):
    # Retrieve the tasks from the ADMET group based on the json
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Filter the ones to run
    data = data[task_group]
    data = {k: v for k, v in data.items()}
    return data


def driver(
    parent_dir,
    model,
    num_options,
    num_walks_per_iter,
    num_iters,
    num_filter_steps,
    parallel_walks,
    beam_widths,
    use_cbr,
    db_path,
    do_pruning,
    use_retrieval,
    save_case,
    replace_cases
):
    # Set up task
    data = get_full_admet_group("adme-tox.json", cfg.TASK_GROUP)
    assert cfg.TASK_NAME in data, f"Provided task name: {cfg.TASK_NAME} - did not match ones in adme-tox.json"

    # Make parent directory where all log files will be stored
    os.makedirs(parent_dir, exist_ok=True)

    print(f"Training for {cfg.TASK_NAME}.....")

    # Extract metric preference (higher or lower metrics used), name of code file, and log folder to use
    metric_dir = data[cfg.TASK_NAME]["Metric"]
    code_name = f"{model}_{cfg.TASK_NAME}.py"
    log_folder = os.path.join(parent_dir, f"{cfg.TASK_NAME}")

    prompt = f"""
    Your task is to write Python code for the following task. The dataset name and task description is provided below.

    Dataset name: {cfg.TASK_NAME}
    Task description: {data[cfg.TASK_NAME]["Description"]}

    Please follow these instructions:
    1. Follow the template in the example code shown below, and do not add any new print statements to it.
    2. Your code must be complete and executable without needing additional user intervention.
    3. Please use Sklearn or Pytorch packages only for your ML implementations.
    \n
    """

    # Example template to prevent TDC API issues during system operation
    template = """Here is an example code snippet showing how to load and evaluate a dataset with the name "Caco2_Wang": 

    from tdc.benchmark_group import admet_group
    group = admet_group(path = 'data/')
    predictions_list = []

    for seed in [1, 2, 3, 4, 5]:
        benchmark = group.get('Caco2_Wang') 
        # all benchmark names in a benchmark group are stored in group.dataset_names
        predictions = {}
        name = benchmark['name']
        train_val, test = benchmark['train_val'], benchmark['test']
        train, valid = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = seed)
        # NOTE: For the dataset, column names are 'Drug' (for the input SMILES strings) and 'Y' (for the output labels)
            
            # --------------------------------------------- # 
            #  Train your model using train, valid, test    #
            #  Save test prediction in y_pred_test variable #
            # --------------------------------------------- #
                    
        predictions[name] = y_pred_test    # For classification tasks, predict probability of positive class
        predictions_list.append(predictions)

    results = group.evaluate_many(predictions_list)
    print(results)
    """

    input_prompt = prompt + template
    print(input_prompt)

    # Run FoO-agentic pipeline with specified inputs and hyperparameters
    output, best_metric, costs = run_pipeline(
        input_prompt,
        code_name,
        log_folder,
        num_options=num_options,
        parallel_walks=parallel_walks,
        num_walks_per_iter=num_walks_per_iter,
        num_iters=num_iters,
        num_filter_steps=num_filter_steps,
        beam_widths=beam_widths,
        metric_dir=metric_dir,
        use_cbr=use_cbr,
        db_path=db_path,
        do_pruning=do_pruning,
        use_retrieval=use_retrieval,
        save_case=save_case,
        replace_cases=replace_cases
    )

    # Print out the results in console
    print(f"Task Name: {cfg.TASK_NAME}")  
    print(output)
    print(f"Cost breakdown: {costs}")

    # Write out final results for viewing in a log file
    with open(f'Results_log.txt', "w") as f:
        f.write(cfg.TASK_NAME)
        f.write("\n")
        for num, outcome in output.items():
            f.write(f"Run {num}: {outcome}")
            f.write("\n")
        f.write(f"COSTS: {costs}")
        f.write("".join(["#"] * 50) + "\n")
    f.close()

    return best_metric


best_output = driver(
    cfg.PARENT_DIR,
    cfg.MODEL,
    cfg.NUM_OPTIONS,
    cfg.NUM_WALKS_PER_ITER,
    cfg.NUM_ITERS,
    cfg.NUM_FILTER_STEPS,
    cfg.PARALLEL_WALKS,
    cfg.BEAM_WIDTHS,
    cfg.USE_CBR,
    cfg.DB_PATH,
    cfg.DO_PRUNING,
    cfg.USE_RETRIEVAL,
    cfg.SAVE_CASES,
    cfg.REPLACE_CASES
)
print("BEST RESULT: ", best_output)
