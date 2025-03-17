# List of hyperparameters (Set to Deployment)
NUM_OPTIONS = 3                          # k: Number of options to generate
NUM_ITERS = 4                            # T: Number of times to iterate values in the FoO
# NUM_WALKS_PER_ITER must be a list of length NUM_ITERS
NUM_WALKS_PER_ITER = [3, 3, 3, 3]        # j: num walks to run in a batch (j) at each iteration
NUM_FILTER_STEPS = 3                     # n: Number of steps to filter task plan to (depth of FoO)
# BEAM_WIDTHS must be a list of length NUM_ITERS
BEAM_WIDTHS = [1.0, 1.0, 0.5, 0.5]       # b: beam width at each iteration as % of top-k: A small % value used to sample top-1
PARALLEL_WALKS = True                    # Whether to execute each set of walks in parallel
MODEL = "gpt-4o-2024-05-13"              # Foundational LLM to use (Recommended LLMs: "gpt-4o" or "gpt-4o-2024-05-13")
USE_CBR = True                           # Whether to use case-based reasoning
DO_PRUNING = True                        # Whether to prune retrieve FoOs (when using CBR)
USE_RETRIEVAL = True                     # Retrieve the closest FoO from the database. Otherwise, the task will be treated as a new case
SAVE_CASES = True                        # Whether to save the cases into the database
REPLACE_CASES = False                    # If saving, whether to overwrite old cases if the new case is similar (prevent DB size from exploding)

# Logging Information
PARENT_DIR = "Log_TDC"                   # Log folder for tracking the run
DB_PATH = "Case-based-foo-TDC"           # Path to case-based database of FoO and tasks (If CBR used)

# ADME-Tox task group and task name
TASK_GROUP = "Absorption"                # "Absorption" "Distribution" "Metabolism" "Excretion" "Toxicity"
TASK_NAME = "Pgp_Broccatelli"            # See adme-tox.json for task names

# Reproducibility (to approximate outcomes). Also triggers LiteLLM caching.
SEED = None                              # Set to None for random seed

# Timeout (in seconds) for code execution. Increase this for tasks such as training ML networks
TIMEOUT = 300                            # 300 seconds sufficient for simpler tabular tasks 

# Model Settings
EMBEDDING_MODEL = "BAAI/llm-embedder"    # HuggingFace embedding model used to compute task similarity for CBR
