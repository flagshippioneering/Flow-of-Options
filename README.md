# Flow-of-Options: Diversified and Improved LLM Reasoning by Thinking Through Options
Flow-of-Options is a network that encapsulates different options for solving a task, as nodes within the network. It expands the scope of the LLM's reasoning, to eliminate pre-training biases in the LLMs' outputs. These biases cause LLMs to always prefer certain approaches over others, and Flow-of-Options is intended to mitigate this bias to improve diversity of the LLM's outputs.

Please refer to our paper for more details: [Flow-of-Options: Diversified and Improved LLM Reasoning by Thinking Through Options](https://arxiv.org/abs/2502.12929)

The code in this repository is an agentic system developed using Flow-of-Options under the hood. In this context, FoO enables the system to explore a range of options for tackling a given problem, e.g., FoO enables the agent to explore a range of ML methods for an ML problem. This repository is setup for the example of solving ADME-Tox tasks from Therapeutic Data Commons ([TDC](https://tdcommons.ai/)).

## Getting started
The system is currently setup to run OpenAI models (See `llm.py`).

1. Create new environment if needed (e.g., using uv): `uv venv <your_env> --python 3.10`
2. Activate environment: `source <your_env>/bin/activate`
3. Setup OpenAI API key as: `export OPENAI_API_KEY=<insert your OpenAI API key>`
4. Unzip this repository into desired folder: `<my_dir>` and `cd <my_dir>`
5. Install requirements as: `uv pip install -r requirements.txt`
6. Run the code using: `python driver.py` -- You can specify the ADME-Tox task to use as `TASK_NAME` and `TASK_GROUP` in `config.py`. Hyperparameters to our system can also be modified in the config file.

**NOTE:** For a custom prompt, modify `input_prompt` passed into `run_pipeline()` in `driver.py`. Please specify in your prompt, either the location of your dataset or how to obtain it (unless it is a well-known example like MNIST or ImageNet for example). For reference, some example prompts are provided in the Appendix section of our paper.

**NOTE:** By default, the config file is set up for development. Please run development prior to running deployment. Deployment assumes that FoO has already been generated and stored in the case database for some tasks.

## Debugging package import errors during execution
During a run, the agent stores useful information in folders labeled `ITERS_WALK_<N>`. The `log.txt` files in these folders will note errors during debugging such as the need for a package that is not installed in the environment. Monitoring these logs will help resolve these package errors which are currently not automatically resolved since we prevent agents from installing packages themselves within the user environment.

### Saved Outputs
The best output code found by the agent is stored under `PARENT_DIR` (from `config.py`) as `BEST_<model_name>_<task_name>.py`, e.g., `BEST_gpt-4o-2024-05-13_Pgp_Broccatelli.py`. The FoO visualizations are saved as `.png` files into `PARENT_DIR`.

### Suggested Configurations
For development, the following are the suggested settings (these are the default settings in the config file):
```
NUM_OPTIONS = 3
NUM_ITERS = 4
NUM_WALKS_PER_ITER = [3, 3, 3, 3]
NUM_FILTER_STEPS = 3
BEAM_WIDTHS = [1.0, 1.0, 0.5, 0.5]
PARALLEL_WALKS = True
MODEL = "gpt-4o-2024-05-13"
USE_CBR = True
DO_PRUNING = True
USE_RETRIEVAL = True
SAVE_CASES = True      # If you'd like to add new cases to the database, set to True
REPLACE_CASES = False  # If True, existing cases in the database will be replaced (e.g., when better FoO found for an existing task)
SEED = None            # Ensuring output diversity
```

For quicker development, you can reduce the number of iterations and walks per iterations. For example:
```
NUM_ITERS = 3
NUM_WALKS_PER_ITER = [3, 3, 2]
BEAM_WIDTHS = [1.0, 1.0, 0.5]
```

For deployment, the following are the suggested settings (assumes development complete):
```
NUM_OPTIONS = 0
NUM_ITERS = 1
NUM_WALKS_PER_ITER = [1]
NUM_FILTER_STEPS = 3
BEAM_WIDTHS = [0.1]
PARALLEL_WALKS = True
MODEL = "gpt-4o-2024-05-13"
USE_CBR = True
DO_PRUNING = True
USE_RETRIEVAL = True
SAVE_CASES = False
REPLACE_CASES = False
```

## License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

## Disclaimer
This is not an officially supported product. It is primarily intended for research and academic purposes only.

## Citation
If you find this repository useful for your work, please consider citing us as follows:
```
@article{nair2025flow,
  title={Flow-of-Options: Diversified and Improved LLM Reasoning by Thinking Through Options},
  author={Nair, Lakshmi and Trase, Ian and Kim, Mark},
  journal={arXiv preprint arXiv:2502.12929},
  year={2025}
}
```
