"""
Module for generating, adapting, executing, and analyzing high-level task plans using Flow-of-Options (FoO)
and case-based reasoning (CBR). This module implements a range of functionalities for constructing, adapting,
and evaluating execution plans for tasks in a structured and iterative manner. The module heavily relies on
integration with large language models (LLMs) to generate detailed plans, simulate potential solutions, and
utilize previously stored case data through CBR for optimized task execution.

Functions:
    generate_high_level_plan(task: str, log_folder: str) -> list[str]:
        Generate a detailed step-by-step task plan using language models and log the plan output.

    filter_task_plan(task: str, plan: str, num_steps: int=2) -> list[str]:
        Filter the task plan steps based on their importance, returning the top-ranked steps.

    generate_adapted_plan_and_options(task, plan=None, options=None):
        Generate adapted plans or options to align with a given task using LLMs.

    generate_nodes(task: str, step: str, prev_steps: list[str], case_opts: list[str], num_nodes: int=3) -> list[Node]:
        Generate diverse implementation nodes for a specific task step, considering previous steps.

    save_foo(foo: FoO, fname: str) -> None:
        Save a Flow-of-Options (FoO) object to a specified file.

    load_foo(fname: str) -> FoO:
        Load a Flow-of-Options object from a file.

    adapt_foo(task: str, foo: FoO) -> FoO:
        Adapt an existing FoO for a new task context, marking nodes as unvisited.

    merge_foo(case_foo: FoO, new_foo: FoO) -> FoO:
        Merge two Flow-of-Options structures into one, maintaining task execution consistency.

    check_if_best(foo: FoO, best_result: float, results: float) -> bool:
        Evaluate if the current results are superior to the best results recorded thus far.

    save_cbr_cases(...)
        Save a case into the CBR database and update it with new information.

    explore_options(...)
        Generate and explore options for different steps in a task plan using LLMs.

    prepare_for_cbr(...)
        Prepare the system for case-based reasoning by initializing the database and retrieving similar cases.

    evaluate_best_walk(...)
        Identify and save the best execution path through multiple iterations of task planning.

    adapt_and_prepare_case_foo(...)
        Adapt and prepare a retrieved FoO from a previous task for application to the current task.

    read_and_save_code(best_code: str, log_folder: str, code_fname: str)
        Read and save the best code execution to the specified log folder.

    base_sanity_checks(beam_widths, num_walks_per_iter, num_iters)
        Perform basic parameter checks to ensure correct configuration.

    run_pipeline(...)
        Execute a complete run of task planning, execution simulation, and analysis with optional CBR.

Use Cases:
    - Create detailed task execution plans and simulations leveraging language models.
    - Integrate prior case information to enhance task execution and learning through CBR.
    - Visualize, analyze, and refine task execution paths using a structured FoO.

Dependencies:
    - Various utility modules and functions for logging, prompt generation, CBR database access, and language model interaction.
    - External libraries: pickle, time, uuid, tqdm for handling data persistence, timing operations, unique identifier generation, and progress tracking.
"""


from __future__ import annotations
import random
import os
import ast
from tqdm import tqdm
from llm import *
from prompts import *
from utils import save_generated_code, log_response
from cbr import Case, CodeDataBase
import pickle
import time
import uuid
from FoO_impl import Node, FoO


def generate_high_level_plan(task: str, log_folder: str) -> list[str]:
	prompt = generate_planner_prompt(task)
	output, cost = call_litellm(prompt)
	log_response(log_folder, f"HIGH LEVEL PLAN: \n {output}")
	return list(ast.literal_eval(output).values()), cost


def filter_task_plan(task: str, plan: str , num_steps: int=2) -> list[str]:
	prompt = generate_rank_steps_prompt(task, plan)  # Rank the steps in the task plan based on importance.
	output, cost = call_litellm(prompt)
	output = ast.literal_eval(output)
	
	# We want to maintain the order of steps in the original plan and pick the top ranked plan steps.
	steps = []
	ranks = [r + 1 for r in range(num_steps)]
	for k, v in output.items():
		if v in ranks:
			steps.append(k)
	
	assert steps  # Ensure that steps extracted
	return steps, cost


def generate_adapted_plan_and_options(task, plan=None, options=None):
	assert plan is not None or options is not None, "Either plan or options must be specified."
	if plan is not None:
		prompt = generate_adapter_prompt(task, plan=plan)
		output, cost = call_litellm(prompt)
		output_plan = ast.literal_eval(output)
		output_plan = list(output_plan.values())  # Adapted task plan
		assert len(output_plan) == len(plan)
		return output_plan, cost

	if options is not None:
		# Adapt options in ranked plan
		prompt = generate_adapter_prompt(task, options=options)
		output, cost = call_litellm(prompt)
		output_options = ast.literal_eval(output)
		output_options = list(output_options.values())
		assert len(output_options) == len(options)
		return output_options, cost


def generate_nodes_helper(
	task: str,
	step: str,
	prev_steps: list[str],
	num_nodes: int,
	case_opts: list[str]=None
) -> list[str]:
	
	if not case_opts:
		prompt = generate_nodes_prompt(task, step, num_nodes, prev_steps)
	else:
		prompt = generate_nodes_given_case_prompt(
			task, step, num_nodes, prev_steps, case_opts
		)
	
	output, cost = call_litellm(prompt, temperature=0.7)
	choices = list(ast.literal_eval(output).values())
	# Filter choices if LLM produced more than the requested number of options
	choices = choices[:num_nodes]
	return choices, cost


def generate_nodes(
	task: str,
	step: str,
	prev_steps: list[str],
	case_opts: list[str],
	num_nodes: int=3
) -> list[Node]:

	nodes, cost = generate_nodes_helper(task, step, prev_steps, num_nodes, case_opts)
	assert isinstance(nodes, list)
	depths = []
	for node in nodes:
		s = Node(task, step, node)
		depths.append(s)

	return depths, cost


def save_foo(foo: FoO, fname: str) -> None:
	# Save the FoO with all visited nodes and corresponding values
	with open(fname, 'wb') as f:
		pickle.dump(foo, f, pickle.HIGHEST_PROTOCOL)


def load_foo(fname: str) -> FoO:
	# Load the FoO from the case
	with open(fname, 'rb') as f:
		foo = pickle.load(f)
	f.close()
	return foo


def adapt_foo(task: str, foo: FoO) -> FoO:
	# 1. Modify all nodes to mark them as unvisited
	# 2. Modify all nodes' string descriptions to match current task
	for s in foo.nodes:
		s.visited_children = {}  # Mark all children as unvisited

	options = [c.code_snippet for c in foo.nodes if c.code_snippet]
	nodes = [c for c in foo.nodes if c.code_snippet]
	options, cost = generate_adapted_plan_and_options(task, options=options)
	assert len(options) == len(nodes)
	for i in range(len(nodes)):
		# Update node descriptions with the adapted options
		nodes[i].code_snippet = options[i]
	return foo, cost


def merge_foo(case_foo: FoO, new_foo: FoO) -> FoO:
	# Assumes that case_foo and new_foo follow a similar plan and have same depth
	# Merges new_foo into case_foo 
	# NOTE: Order matters to retain case values at the root without needing additional manipulation
	assert case_foo.depth == new_foo.depth, "FoOs must have same depth for merging."
	parents = [case_foo.root]
	foo1_ptr = case_foo.root.get_next_nodes()
	foo2_ptr = new_foo.root.get_next_nodes()
	for _ in range(case_foo.depth):
		for parent in parents:
			children = foo1_ptr + foo2_ptr
			for child in children:
				child.add_parent(parent)
				parent.add_child(child)
				# Add foo2's nodes into nodes of foo1
				case_foo.add_nodes(child)
		foo1_ptr = foo1_ptr[-1].get_next_nodes()
		foo2_ptr = foo2_ptr[-1].get_next_nodes()
		parents = children

	# Remove root of foo_2 from parents of foo_1
	for s1 in case_foo.nodes.keys():
		if new_foo.root in s1.get_parents():
			s1.remove_parent(new_foo.root)

	# Merge valid and invalid paths from both FoOs
	case_foo.valid_paths += new_foo.valid_paths
	case_foo.invalid_paths += new_foo.invalid_paths

	return case_foo


def check_if_best(foo: FoO, best_result: float, results: float) -> bool:
	# A way to track the best output seen so far
	if "high" in foo.metric_dir.lower():
		if best_result < results:
			return True
	else:
		if best_result > results:
			return True
	return False


def save_cbr_cases(
	db_path: str,
	db: CodeDataBase,
	task_brief: str,
	plan: str,
	ranked_plan: list[str],
	best_result: float,
	foo: FoO,
	closest_case: Case,
	similarity: float
) -> None:
	"""
	Save a case into the CodeDataBase and update the database with the new information.

	This function handles the process of saving a new or updated case into a case-based reasoning (CBR) database.
	It generates unique identifiers for cases, saves related Flow-of-Options (FoO) objects, updates the existing
	case database, and ensures changes are persisted as a JSON file.

	Args:
		db_path (str): Path to the database where cases are stored.
		db (CodeDataBase): The existing database object where cases are maintained.
		task_brief (str): A brief description of the task for which the case is being saved.
		plan (str): The high-level plan associated with the case.
		ranked_plan (list[str]): A list of steps from the plan ranked by importance.
		best_result (float): The best result metric obtained from the task execution.
		foo (FoO): The Flow-of-Options object representing task execution paths.
		closest_case (Case): The most similar existing case in the database to the current one.
		similarity (float): The similarity score between the current task and the closest existing case.

	Returns:
		None

	Side Effects:
		- Generates a unique identifier for the new case.
		- Saves the FoO to a file in the specified database path.
		- Updates and saves the database as a JSON file for persistence.
	"""

	# Save case into database
	id = uuid.uuid4()  # Generate a unique ID
	foo_fname = os.path.join(db_path, f"{id}_foo.pkl")
	case = Case(
		task=task_brief,
		plan=plan,
		ranked_plan=ranked_plan,
		foo=foo_fname,
		best_metric=best_result,
		metric_dir=foo.metric_dir
	)
	new_case = db.add_case(case, closest_case, similarity)    # Update the db
	# If replacing an existing case, replace the old FoO to keep relevant pkl files only
	# If adding as a new case, save as a new FoO in a new pkl file
	save_foo(foo, closest_case.foo if not new_case else foo_fname)
	db.save_into_json()  # Save new DB as json


def explore_options(
	num_options: int, task: str, ranked_plan: list[str], foo_opts: list[list[str]]
) -> tuple[list, float]:
	"""
	Function to explore options for different steps in a task plan, using LLMs.

	Args:
		num_options (int): Number of options to explore for each step in the task plan.
		task (str): The task specified by the user.
		ranked_plan (list): List of most important steps in the task plan to generate options for.
		foo_opts (list): List of past explored options (if obtained from closest case via CBR).
			If supplied, new options generated will be conditioned on past explored options.

	Returns:
		all_walks_as_nodes (list): A list of options (specified as node objects) for steps in task plan.
		option_costs (float): Total query cost of generating these options using LLMs.
	"""

	all_walks = [[] for _ in range(num_options)]
	all_walks_as_nodes = [[] for _ in range(num_options)]
	option_costs = 0.0

	for step_id, step in enumerate(tqdm(ranked_plan)):
		print(f"Step {step_id + 1} of filtered plan: {step}")
		for walk_num in range(num_options):
			print("Generating nodes (options)...")
			case_opts = [] if not foo_opts else foo_opts[step_id]
			nodes, cost = generate_nodes(task, step, all_walks[walk_num], case_opts)
			option_costs += cost

			# Randomly select one of the nodes
			s = random.choices(nodes, k=1)[0]

			chosen_node = s.code_snippet
			all_walks[walk_num].append(f"Implementation detail {step_id}: {chosen_node}\n")
			all_walks_as_nodes[walk_num].append(s)
	
	return all_walks_as_nodes, option_costs


def prepare_for_cbr(
	task_brief: str,
	db_path: str,
	replace_cases: bool,
	use_retrieval: bool
) -> tuple[CodeDataBase, Case, float]:
	"""Function to prepare system to perform case-based reasoning.

	Args:
		task_brief (str): Brief description of user task.
		db_path (str): Path to the case-based reasoning database.
		replace_cases (bool): Whether to replace highly similar cases within the database (possibly overwriting past cases) when saving.
		use_retrieval (bool): Whether to retrieve closest case to reuse FoO or just treat the case as a separate one.

	Returns:
		db (CodeDatabase): Database storing the cases.
		closest_case (Case): The most similar existing case in the database to the current one.
		similarity (float): The similarity score between the current task and the closest existing case.
	"""
	
	# Initiation of database
	print("Using case-based reasoning...")
	assert db_path
	db = CodeDataBase(db_path, replace_cases=replace_cases)
	os.makedirs(db_path, exist_ok=True)
	# Initialize empty case and no similarity
	similarity = 0.0
	closest_case = None
	if use_retrieval:
		# Retrieve the closest case if it exists
		closest_case, similarity = db.retrieve_case_given_task(task_brief, exact_match=False)
		if closest_case is not None and similarity < 0.7:
			# Set a threshold on similarity / relevance
			closest_case = None
	else:
		db = None
		closest_case = None

	return db, closest_case, similarity


def evaluate_best_walk(
	results: dict, best_result: float, best_code: str, code_names: list, foo: FoO
) -> tuple[float, str]:
	"""Evaluate the best walk given a set of iterations.

	Args:
		results (dict): Mapping denoting results obtained at each iteration.
		best_result (float): Best result obtained thus far.
		best_code (str): Best Python script (it will correspond to the best result).
		code_names (list): List of Python script names for each iteration.
		foo (FoO): Flow-of-Options structure observed thus far.

	Returns:
		best_result (float): Best result obtained thus far.
		best_code (str): Python script name corresponding to the best result.
	"""

	for walk_num in results.keys():
		if best_result is None:
			best_result = results[walk_num]
			best_code = code_names[walk_num]
		else:
			if check_if_best(foo, best_result, results[walk_num]):
				best_result = results[walk_num]
				best_code = code_names[walk_num]
	return best_result, best_code


def adapt_and_prepare_case_foo(
	task: str,
	closest_case: Case,
	metric_dir: str,
	costs: dict,
	log_folder: str,
	do_pruning: bool,
	foo_viz_fname: str
):
	"""
	Function for preparing and adapting a retrieved case from a previous task, to the current task.

	Args:
		task (str): Task specified by the user.
		closest_case (Case): The most similar existing case in the database to the current one.
		metric_dir (str): Metric direction to use for the task (whether higher or lower metrics preferred).
		costs (dict): Cost breakdown mapping LLM operations to corresponding costs.
		log_folder (str): Log folder for storing FoO visualizations.
		do_pruning (bool): Whether to prune the FoO.
		foo_viz_fname (str): Name of the file to save FoO visualization.

	Returns:
		adapted_case_foo (FoO): The case adapted to the new task.
		adapted_plan (str): Task plan adapted from previous task to the new task.
		adapted_ranked_plan (list): Important plan steps adapted from previous task to new task.
		costs (dict): Cost breakdown with the case adaptation costs.
	"""

	print("Pruning and preparing the retrieved FoO, including visualization...")
	# Retrieve plan, and FoO from closest case
	plan = closest_case.plan
	ranked_plan = closest_case.ranked_plan
	case_foo = load_foo(closest_case.foo)
	case_foo.metric_dir = metric_dir  # Keep the current task's metric
	
	if do_pruning:
		case_foo.prune(num_top=2)  # Keep top two branches only
		case_foo.visualize_foo(os.path.join(log_folder, "PRUNED_" + foo_viz_fname))

	# Adapt FoO to new case
	adapted_case_foo, cost = adapt_foo(task, case_foo)
	costs["foo-adapt"] = cost

	# Adapt task plan to new case
	adapted_plan, cost = generate_adapted_plan_and_options(task, plan)
	costs["high-level-plan"] = cost
	
	# Adapt selected implementation steps to new case
	adapted_ranked_plan, cost = generate_adapted_plan_and_options(task, ranked_plan)
	costs["plan-rank"] = cost
	return adapted_case_foo, adapted_plan, adapted_ranked_plan, costs


def read_and_save_code(best_code: str, log_folder: str, code_fname: str) -> None:
	# Read and re-save the best code in the log folder
	with open(best_code, "r") as f:
		code = f.read()
	f.close()
	save_generated_code(log_folder, "BEST_" + code_fname, code)


def base_sanity_checks(beam_widths, num_walks_per_iter, num_iters):
	# Some simple sanity checks to ensure that parameters are correctly specified.
	if isinstance(beam_widths, list):
		assert len(beam_widths) == num_iters, "One top-k value for each iteration must be specified."
	else:
		beam_widths = [beam_widths] * num_iters

	if isinstance(num_walks_per_iter, list):
		assert len(num_walks_per_iter) == num_iters
	else:
		num_walks_per_iter = [num_walks_per_iter] * num_iters


def run_pipeline(
	task: str,
	code_fname: str,
	log_folder: str,
	num_options: int,
	num_walks_per_iter: list[int] | int,
	num_iters: int=3,
	num_filter_steps: int=2,
	parallel_walks: bool=False,
	foo_viz_fname: str="FoO.png",
	beam_widths: list[float] | float=0.5,
	metric_dir: str=None,
	use_cbr: bool=False,
	db_path: str=None,
	do_pruning: bool=False,
	use_retrieval: bool=False,
	save_case: bool=False,
	replace_cases: bool=False
) -> tuple[dict, float, dict]:
	"""
	Function that generates high-level task plans, simulates task execution using Flow-of-Options, and
	analyzes the execution using case-based reasoning.

	Args:
		task (str): User prompt with relevant information about task,
		code_fname (str): Name of Python file for saving the output code,
		log_folder (str): Name of the log folder for saving execution outputs,
		num_options (int): Number of options to generate with LLM
		num_walks_per_iter (list): Number of walks through the FoO before value updates happen (similar to batch size) 
		num_iters (int): Number of update iterations of the FoO. Default is 3.
		num_filter_steps (int): Number of important steps to filter the task plan into. This is the depth of the FoO.
		parallel_walks (bool): Whether to run the walks within each update iteration in parallel. Default is False.
		foo_viz_fname (str): Filename for saving a visualization of the Flow-of-Options. Default is FoO.png.
		beam_widths (list): Beam widths to use within each update iteration.
		metric_dir (str): Whether higher or lower metric is preferred. If unspecified, an LLM is used to make a guess.
		use_cbr (bool): Whether to use case-based reasoning.
		db_path (str): Path to the case-based reasoning database.
		do_pruning (bool): Whether to prune the case-based retrieved FoO.
		use_retrieval (bool): Whether to retrieve closest case to reuse FoO or just treat the case as a separate one.
		save_case (bool): Whether to save the code executions into the case bank.
		replace_cases (bool): Whether to replace highly similar cases within the database (possibly overwriting past cases) when saving.

	Returns:
		iter_results (dict): A dictionary mapping each update iteration to the best outcome metric from that iteration.
		best_result (float): The best overall metric obtained from running the full set of iterations.
		costs (dict): A dictionary mapping cost breakdown of each step to the cost.
	"""

	# Base sanity checks on whether parameters are correctly specified.
	base_sanity_checks(beam_widths, num_walks_per_iter, num_iters)

	# Generate a brief description of the task that we can use for retrieval and for saving cases in CBR.
	task_brief = call_model_openai(f"Please generate a summary of the task description: {task}")[0]

	# Track best result and best code
	best_result = None
	best_code = None
	costs = {}

	# Case-based reasoning
	if use_cbr:
		db, closest_case, similarity = prepare_for_cbr(task_brief, db_path, replace_cases, use_retrieval)

	os.makedirs(log_folder, exist_ok=True)
	log_response(log_folder, task, new_file=True)

	if closest_case is not None:
		case_foo, plan, ranked_plan, costs = adapt_and_prepare_case_foo(
			task, closest_case, metric_dir, costs, log_folder, do_pruning, foo_viz_fname
		)

		# Run verification on the retrieved paths from previous case (deployment)
		results, code_names, _ = case_foo.run_iters(
			task,
			plan,
			log_folder,
			code_fname,
			parallel_walks,
			num_walks_per_iter=1,
			beam_width=1.0,
			reuse_paths=case_foo.valid_paths
		)

		# Save best code to the main parent log folder and note the best result
		best_result, best_code = evaluate_best_walk(results, best_result, best_code, code_names, case_foo)
		read_and_save_code(best_code, log_folder, code_fname)

		# Extract options as strings so that new LLM option generations can be conditioned on them.
		foo_opts = case_foo.get_foo_as_options()
	else:
		print("Generating high-level task plan...")
		plan, cost = generate_high_level_plan(task, log_folder)
		costs["high-level-plan"] = cost
		ranked_plan, cost = filter_task_plan(task, plan, num_steps=num_filter_steps)
		costs["plan-rank"] = cost
		case_foo = None  # No cases retrieved
		foo_opts = []
	
	# Generation of options (development)
	print(f"Generating options.... Num Options: {num_options}")
	all_walks_as_nodes, option_costs = explore_options(num_options, task, ranked_plan, foo_opts)

	# Track option generation costs
	costs["option-generation"] = option_costs
	iter_results = {}

	# Create a FoO of the initial random walks and assign metric direction if available
	if num_options != 0:
		foo = FoO(all_walks_as_nodes, metric_dir=metric_dir)
		if closest_case is not None:
			print("Merging case FoO and new FoO...")
			# Merge new FoO into the case FoO and retain case values from before
			foo = merge_foo(case_foo, foo)
	else:
		assert case_foo is not None, "Case bank empty, and option generation disabled (num_options = 0). No FoO constructed."
		foo = case_foo
	
	# Visualize starting FoO
	foo.visualize_foo(os.path.join(log_folder, "START_" + foo_viz_fname))

	start = time.time()
	# Run walks on the FoO to compute values and find a good solution
	walk_cost = {k: 0.0 for k in range(num_iters)}
	for iter in range(num_iters):
		print(f"Running Iter {iter}...")
		results, code_names, cost = foo.run_iters(
			task,
			plan,
			log_folder,
			code_fname,
			parallel_walks,
			num_walks_per_iter=num_walks_per_iter[iter],
			beam_width=beam_widths[iter]
		)

		# Track results at every iteration
		iter_results[iter] = results
		walk_cost[iter] += cost

		# Save best code to the main parent log folder and note the best result
		best_result, best_code = evaluate_best_walk(results, best_result, best_code, code_names, foo)
		read_and_save_code(best_code, log_folder, code_fname)

		print(f"*** Best output metric obtained so far ***: {best_result}")

	# Track walks cost
	costs["Costs-per-iter"] = walk_cost

	# Visualize the generated FoO
	foo.visualize_foo(os.path.join(log_folder, foo_viz_fname))
	log_response(log_folder, f"Walk Results: \n{iter_results}")
	log_response(log_folder, f"All costs: \n{costs}")

	# Case-based reasoning (save cases)
	if db is not None and save_case:
		save_cbr_cases(db_path, db, task_brief, plan, ranked_plan, best_result, foo, closest_case, similarity)

	end = time.time()

	# Get sum of all costs
	sum_c = 0.0
	for c in costs.values():
		sum_c += sum(c.values()) if isinstance(c, dict) else c
	log_response(log_folder, f"Total costs: {sum_c}")
	print("Total cost: ", sum_c)
	print("Total time (seconds): ", end - start)
	return iter_results, best_result, costs
