"""Module for constructing and exploring a Flow-of-Options (FoO) structure.

This module provides classes and methods to construct a graph-like structure termed as 
Flow-of-Options (FoO), which can be used to model task-solving strategies or plans 
as sequences of steps. Each step is a node representing a task component, 
and edges indicate possible transitions with their associated values. Values for each
edge denotes the final metric obtained when the task is executed using the steps
represented by the corresponding nodes along that edge.

Classes:
    Node: Represents a single task component within a FoO. Nodes contain information 
    about the task, plan step represented by the node, code snippet denoting the
	implementation of the step, connections to children nodes, parents nodes, and
	value assignments.

    FoO: Encapsulates the FoO structure, containing methods for:
        - Initializing the structure through random walks.
        - Performing consistency checks and pruning strategies.
        - Running iterative simulations to evaluate paths.
        - Simulating paths both exhaustively and probabilistically through beam search.
        - Visualizing the graph structure.

Functions:
    background(f): Decorator to perform functions in the background using asyncio.
    run_parallel_walks_execution(implementation, walk_log_folder, walk_num, code_fname, task, plan, reused) -> dict | str:
        Executes task implementations in parallel via an event loop.

    run_walk(implementation, walk_log_folder, walk_num, code_fname, task, plan, reused=False) -> tuple[dict | str, float]:
        Executes a single walk through the FoO, generating and debugging the resulting code.

Use Cases:
    - Constructing task solution strategies using interconnected nodes.
    - Simulating potential pathways through the solution space to identify optimal strategies.
    - Visualizing and managing node structures to optimize task flows.
    - Running parallel executions of task implementation to enhance efficiency.

Imported Modules:
    This module imports functionality for language models, data processing, 
    graph generation, and parallel asynchronous execution. Specific utilities handle
    tasks such as log responses, code generation, and result summarization.
"""


from __future__ import annotations
from llm import *
from prompts import *
from utils import create_foo_graph
import random
import asyncio
import math
import heapq


class Node():
	# Class denoting nodes in the FoO
	def __init__(self, task: str, plan_step: str, code_snippet: str) -> None:
		self.task = task
		self.plan_step = plan_step
		self.code_snippet = code_snippet
		self.__values = {}          # Maps from child nodes to a value for the self -> child connection.
		self.children = {}          # Child nodes
		self.parents = {}           # Parents of self
		self.visited_children = {}  # All visited children of self

	def set_value(self, child: Node, value: float) -> None:
		# Add value from self to its child.
		self.__values[child] = value

	def get_value(self, child: Node) -> None | float:
		if self.children:
			return self.__values[child]

	def add_child(self, child: Node, value: float=-1000.0) -> None:
		# Values initializes to a low starting value
		snippets = [c.code_snippet for c in self.children]
		if child not in self.children and child.code_snippet not in snippets:
			self.children[child] = True
			self.set_value(child, value)

	def add_children(self, children: list[Node]) -> None:
		for child in children:
			self.add_child(child)

	def add_parent(self, parent: Node) -> None:
		snippets = [p.code_snippet for p in self.parents]
		if parent not in self.parents and parent.code_snippet not in snippets:
			self.parents[parent] = True

	def add_parents(self, parents: list[Node]) -> None:
		for parent in parents:
			self.add_parent(parent)

	def check_if_child_visited(self, child: Node) -> bool:
		return True if child in self.visited_children else False
	
	def add_child_visited(self, child: Node) -> None:
		# Function that marks a child node as visited
		self.visited_children[child] = True

	def get_next_nodes(self) -> list[Node]:
		return list(self.children.keys())
	
	def get_parents(self) -> list[Node]:
		return list(self.parents.keys())
	
	def remove_parent(self, p: Node) -> None:
		self.parents.pop(p, None)

	def remove_child(self, c: Node) -> None:
		self.children.pop(c, None)
		self.visited_children.pop(c, None)
		self.__values.pop(c, None)


class FoO():
	def __init__(self, all_walks: list[list[Node]], metric_dir: str) -> None:
		# This builds a FoO from a set of initial walks
		self.root = Node("", "", "")    # NULL root
		self.nodes = {}
		self.nodes[self.root] = True    # Add root to our list
		self._build_foo(all_walks)    	# Initialize FoO
		self.metric_dir = metric_dir    # Decide if "higher" or "lower" values are better
		self.invalid_paths = []
		self.depth = len(all_walks[0])  # Number of nodes within a path
		self.valid_paths = all_walks    # Store valid paths in case consistency check fails

	def _build_foo(self, all_walks: list[list[Node]]) -> None:
		print("Building foo...")
		# Build a foo out of random walks
		paired_walks = list(zip(*all_walks))
		# Add children to root
		self.root.add_children(paired_walks[0])
		# Add root as parent
		for s in paired_walks[0]:
			s.add_parent(self.root)

		for i in range(len(paired_walks)):
			for node in paired_walks[i]:
				if i < len(paired_walks) - 1:
					# Last layer of nodes will have no children.
					node.add_children(paired_walks[i + 1])
				if i > 0:
					node.add_parents(paired_walks[i - 1])
				# Add all nodes to the foo
				self.nodes[node] = True

	def _check_consistent(self, path: list[Node], child: Node) -> bool:
		# Function that checks if a path is valid
		# NOTE: This does occasionally fail with some small probability
		path_as_string = [f"{i}. {s.code_snippet}\n" for i, s in enumerate(path)]
		path_as_string = "".join(path_as_string)
		prompt = generate_check_consistency_prompt(path_as_string, child.code_snippet)
		output, _ = call_litellm(prompt)
		output = ast.literal_eval(output)
		return "no" in output["contradiction?"].lower()

	def _bfs_all_walks(self) -> list[list[Node]]:
		# Generate all possible "valid" paths through the foo
		# NOTE: This is exhaustive and can be time consuming.
		paths = [[s] for s in self.root.get_next_nodes()]
		new_paths = [[s] for s in self.root.get_next_nodes()]
		assert new_paths, "No starting nodes found!"
		while new_paths:
			new_paths = []
			for path in paths:
				for child in path[-1].get_next_nodes():
					if self._check_consistent(path, child):
						new_paths.append(path + [child])
					else:
						# Track invalid paths
						self.invalid_paths.append(path + [child])
			if new_paths:
				paths = new_paths

		return paths


	def _harmonize_values(self, weights: list[float], beam_width: float) -> list[float]:
		# Function that identifies nodes to explore in beam-search by updating its weights
		# Beam width is the percentage of top cases to explore

		epsilon = 1e-5  # Keep values in bound and avoid zeros
		min_weight = min(weights)
		if min_weight < 0:
			# Move all weights to [0, 0+] direction
			weights = [(w - min_weight) + epsilon for w in weights]

		def keep_top_k(lst, num):
			# Sets all but the top k values in a list to zero
			top_indices = heapq.nlargest(num, range(len(lst)), key=lst.__getitem__)
			result = [0.0] * len(lst)

			for i in top_indices:
				result[i] = 1.0

			return result

		# Apply correction to weights to sample only top k%
		num_cases = math.ceil(len(weights) * beam_width)
		weights = keep_top_k(weights, num_cases)
		# Normalize weights to uniform so all selected top nodes are equally explored
		sum_w = sum(weights)
		weights = [w / sum_w for w in weights]
		return weights


	def _bfs_random_walk(self, beam_width: float) -> list[Node]:
		# Returns one random path through the foo
		# It samples nodes proportional to their weights
		weights = [self.root.get_value(r) for r in self.root.get_next_nodes()]
		weights = self._harmonize_values(weights, beam_width)
		root = random.choices(self.root.get_next_nodes(), weights=weights, k=1)[0]
		
		frontier = [root]
		path = [root]
		while frontier:
			curr_node = frontier.pop()
			valid_children = []
			for child in curr_node.get_next_nodes():
				if self._check_consistent(path, child):
					valid_children.append(child)
				else:
					self.invalid_paths.append(path + [child])
			
			if valid_children:
				# Sample from valid children (to generate coherent paths)
				weights = [curr_node.get_value(c) for c in valid_children]
				weights = self._harmonize_values(weights, beam_width)
				next_child = random.choices(valid_children, weights=weights, k=1)[0]

				path.append(next_child)
				frontier = [next_child]

		if len(path) < self.depth or path in self.invalid_paths:
			# If consistency check somehow failed, return one of the paths we know is valid.
			path = random.choices(self.valid_paths, k=1)[0]

		assert len(path) == self.depth, "Path does not have expected depth."
		return path

	def _update_value(self, paths: list[list[Node]], adjusted_values: dict) -> None:
		# V(s_t, a_t) = max_(a)(s_{t+1}, a)  # Update path values to optimal value
		for walk_num in adjusted_values.keys():
			# Start at the NULL root
			path = [self.root] + paths[walk_num]
			for i in range(len(path) - 1):
				if path[i].check_if_child_visited(path[i + 1]):
					# Child already visited -- update value to optimal
					path[i].set_value(path[i + 1], max(path[i].get_value(path[i + 1]), adjusted_values[walk_num]))
				else:
					# Child not visited. Just update its value with measured value
					path[i].set_value(path[i + 1], adjusted_values[walk_num])
				
				# Update visited children
				path[i].add_child_visited(path[i + 1])
		print("Value updates in FoO complete!")

	def _convert_path_nodes_to_strings(self, path: list[Node]) -> str:
		string_path = []
		for step_id, node in enumerate(path):
			string_path.append(f"Implementation detail {step_id}: {node.code_snippet}\n")
		return string_path

	def _remove_node(self, node: Node) -> None:
		for s in self.nodes.keys():
			if node in s.children:
				# Remove node as child
				s.remove_child(node)
			
			if node in s.parents:
				# Remove node as parent
				s.remove_parent(node)

		# Remove from foo
		self.nodes.pop(node, None)

	def add_nodes(self, node: Node) -> None:
		snippets = [s.code_snippet for s in self.nodes]
		if node not in self.nodes and node.code_snippet not in snippets:
			self.nodes[node] = True

	def run_iters(
		self,
		task: str,
		plan: str,
		log_folder: str,
		code_fname: str,
		parallel_walks: bool,
		num_walks_per_iter: int,
		beam_width: float,
		all_paths: bool=False,
		reuse_paths: list[list[Node]]=[]
	) -> tuple[dict, dict, float]:
		"""Function that performs a walk through a generated flow of options.
		
		Args:
			task (str): Specification of the task
			plan (str): Specification of the high-level task plan
			log_folder (str): Name of the log folder to save run details to.
			code_fname (str): Name of the Python code to save output into.
			parallel_walks (bool): Whether to run all the different code executions in parallel.
			num_walks_per_iter (int): Number of random walks to run per iteration (before value update occurs).
			beam_width (float): Percentage of top nodes to use on each iteration (width of beam-search).
			all_paths (bool): Whether to explore all the possible valid paths when traversing the foo.
			reuse_paths (list): A list of pre-specified paths to run in the iteration instead of exploring new ones.

		Returns:
			values (dict): A mapping from iteration to the corresponding final metric of that iteration.
			code_names (dict): A mapping from iteration to file names of the corresponding produced code.
			total_cost (float): Total cost of running all the iterations.
		"""
		paths = self._generate_path(num_walks_per_iter, beam_width, all_paths=all_paths) if not reuse_paths else reuse_paths
		paths_as_strings = [self._convert_path_nodes_to_strings(p) for p in paths]
		values, code_names, cost, valid_walks = self._simulate_walks(
			task, plan, paths_as_strings, log_folder, code_fname, parallel_walks, bool(reuse_paths)
		)

		# Update valid walks with newly found valid walks
		for walk_num in valid_walks:
			if paths[walk_num] not in self.valid_paths:
				self.valid_paths.append(paths[walk_num])

		# Adjust values to minimize or maximize values depending on regression or classification.
		# Since we're propagating outcome of running code, a lower score is preferred for regression.
		scale = -1 if "low" in self.metric_dir.lower() else 1
		adjusted_values = {k: scale * v for k, v in values.items()}

		if len(paths) < (num_walks_per_iter // 3):
			# Log some error details
			log_response(log_folder, f"BOOTSTRAPPED PATHS STRINGS:\n{paths_as_strings}")
			log_response(log_folder, f"OUTPUT VALUES COMPUTED:\n{adjusted_values}")
			self.visualize_foo("Debug_graph.png")
			raise ValueError("Most of the paths not simulatable. Something is wrong!")
		
		# Update values based on bootstrapped paths based on adjusted values.
		self._update_value(paths, adjusted_values)

		# Return actual code metrics (not adjusted values) and code script names for logging
		return values, code_names, cost

	def _simulate_walks(
		self,
		task: str,
		plan: str,
		implementations: list[list[str]],
		log_folder: str,
		code_fname: str,
		parallel_walks: bool,
		reused: bool=False
	) -> tuple[dict, list[str]]:
		# Create separate folders for the logs from the walks
		walk_folders = {}
		for walk_num in range(len(implementations)):
			walk_log_folder = os.path.join(log_folder, f"ITER_WALK_{walk_num}")
			os.makedirs(walk_log_folder, exist_ok=True)
			walk_folders[walk_num] = walk_log_folder

		results = {}
		code_names = {}
		valid_walks = []  # Indices of the valid walks that produced code
		if parallel_walks:
			# Do all the walks in parallel
			loop = asyncio.get_event_loop()
			looper = asyncio.gather(*[run_parallel_walks_execution(
				implementation,
				walk_folders[walk_num],
				walk_num,
				f"Walk_{walk_num}_{code_fname}",
				task,
				plan,
				reused
			) for walk_num, implementation in enumerate(implementations)])
			outputs = loop.run_until_complete(looper)
			total_cost = sum([c[1] for c in outputs])
		else:
			# Do the walks sequentially
			outputs = []
			total_cost = 0.0
			for walk_num, implementation in enumerate(implementations):
				code_name_with_walk = f"Walk_{walk_num}_{code_fname}"
				result, cost = run_walk(
					implementation,
					walk_folders[walk_num],
					walk_num,
					code_name_with_walk,
					task,
					plan,
					reused=reused
				)
				outputs.append(result)
				total_cost += cost

		# Compile scores
		for walk_num, result in enumerate(outputs):
			if "Error" in result:
				continue
			metric, metric_dir = query_metric_summarizer(task, result, self.metric_dir is None)
			results[walk_num] = float(metric)
			code_names[walk_num] = os.path.join(walk_folders[walk_num], f"Walk_{walk_num}_{code_fname}")
			valid_walks.append(walk_num)

		self.metric_dir = metric_dir if self.metric_dir is None else self.metric_dir
		return results, code_names, total_cost, valid_walks

	def _generate_path(self, num_paths: int, beam_width: float, all_paths: bool=False) -> list[list[Node]]:
		# Pick a root node and expand
		print("Sampling walks...")
		generated_paths = []
		if not all_paths:
			for _ in range(num_paths):
				path = self._bfs_random_walk(beam_width)
				generated_paths.append(path)
		else:
			paths = self._bfs_all_walks()
			generated_paths = paths
		
		assert generated_paths  # Should not be empty or something is wrong.
		if not all_paths:
			assert len(generated_paths) == num_paths, "Enough paths not generated."  # Should generate num_paths otherwise.
		return generated_paths

	def visualize_foo(self, fname: str, with_invalid_paths: bool=True) -> None:
		# Visualize the invalid paths as well.
		if with_invalid_paths:
			invalid_paths = [[s.code_snippet for s in p] for p in self.invalid_paths]
		else:
			invalid_paths = []

		nodes_visited = {}
		graph = {}
		for node in self.nodes:
			graph[node.code_snippet] = []
			for child in node.get_next_nodes():
				graph[node.code_snippet].append((child.code_snippet, node.get_value(child)))

				if child.code_snippet not in nodes_visited:
					nodes_visited[child.code_snippet] = node.check_if_child_visited(child)
				else:
					nodes_visited[child.code_snippet] = nodes_visited[child.code_snippet] or node.check_if_child_visited(child)

		# Add root
		nodes_visited[self.root.code_snippet] = True
		create_foo_graph(graph, nodes_visited, fname, invalid_paths)
		
	def get_foo_as_options(self) -> list[list[str]]:
		# Return a foo as list of strings denoting all of its nodes, separated by depth
		depth = self.depth
		options = []
		curr_node = self.root
		for _ in range(depth):
			options.append([c.code_snippet for c in curr_node.get_next_nodes()])
			curr_node = curr_node.get_next_nodes()[-1]
		return options

	def prune(self, num_top: int=1) -> None:
		"""Function that prunes certain edges of the Flow-of-Options. The edges to be pruned
		are determined based on the values associated with them. Only the top (high-scoring)
		edges are kept in the FoO.
		
		Args:
			num_top (int): Number of top scoring branches to keep when pruning the FoO.

		Returns:
			None.
		"""

		curr_node = self.root
		nodes_to_prune = []  # Nodes to prune in foo
		for _ in range(self.depth):
			child_values = []
			for child in curr_node.get_next_nodes():
				max_value = -100000.0
				for parent in child.get_parents():
					# Get best path value from parent to child
					max_value = max(parent.get_value(child), max_value)
				child_values.append(max_value)

			top_indices = heapq.nlargest(num_top, range(len(child_values)), key=child_values.__getitem__)
			
			for i in range(len(child_values)):
				if i not in top_indices:
					nodes_to_prune.append(curr_node.get_next_nodes()[i])
					
			curr_node = curr_node.get_next_nodes()[-1]

		# Prune the nodes
		for prune_s in nodes_to_prune:
			self._remove_node(prune_s)

		# Prune the nodes from invalid paths by removing paths with the pruned node
		new_invalid_paths = []
		for path in self.invalid_paths:
			path_valid = True
			for prune_s in nodes_to_prune:
				if prune_s in path:
					path_valid = False
					break
			
			if path_valid:
				new_invalid_paths.append(path)

		# Prune the nodes from valid paths also
		new_valid_paths = []
		for path in self.valid_paths:
			path_valid = True
			for prune_s in nodes_to_prune:
				if prune_s in path:
					path_valid = False
					break
			
			if path_valid:
				new_valid_paths.append(path)

		self.invalid_paths = new_invalid_paths
		self.valid_paths = new_valid_paths


# Parallelize the different random walks.
def background(f):
	def wrapped(*args, **kwargs):
		return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
	return wrapped


@background
def run_parallel_walks_execution(
	implementation: list[str],
	walk_log_folder: str,
	walk_num: int,
	code_fname: str,
	task: str,
	plan: str,
	reused: bool
) -> dict | str:

	return run_walk(
		implementation,
		walk_log_folder,
		walk_num,
		code_fname,
		task,
		plan,
		reused
	)


def run_walk(
	implementation: list[str],
	walk_log_folder: str,
	walk_num: int,
	code_fname: str,
	task: str,
	plan: str,
	reused: bool=False
) -> tuple[dict | str, float]:
	# Stitch a plan based on implementation steps, generate and execute code with debugging.
	s = "".join(implementation)
	log_response(walk_log_folder, f"Finalized Implementation {walk_num}:\n{s}", new_file=True)

	if not reused:
		print(f"Constructing and executing code for Walk {walk_num}...")
	code, cost = generate_code(task, plan, implementation, walk_log_folder, code_fname)
	_, _, _, result, return_code = query_debugger(task, code, walk_log_folder, code_fname, walk_num=walk_num)
	if not reused:
		print(f"Completed execution of Walk {walk_num} code")

	# Handle errors if any
	if return_code != 0 or "error" in result.lower() or "exception" in result.lower():
		result = "Error"

	return (result, cost)
