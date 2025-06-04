"""A module for interacting with OpenAI models to assist in agentic tasks.

This module provides functions to preformat input, query a programmer LLM
for code generation based on a prompt, summarize task metrics, debug code,
and generate code by stitching functions. It uses the OpenAI API and LiteLLM
for model interactions.

Functions:
    preformat_input(input: str | list[str]) -> list[dict[str, str]]:
        Formats a string or list of strings into a list of dictionaries
        for LLM input.

    query_programmer(input: str, do_cache: bool = cfg.SEED is not None) -> tuple[dict, float]:
        Queries an LLM to generate code for a given input.

    query_metric_summarizer(task: str, result: str, get_metric: bool = False) -> tuple[float, str]:
        Summarizes the metrics of a task based on its results using an LLM.

    query_debugger(task: str, code: str, log_folder: str, code_fname: str, max_error_iterations: int = 5, walk_num: int = None) -> tuple[str, int, int, str, int]:
        Debugs provided code by iteratively querying a LLM until the code runs without errors.

    plan_stitch(task: str, plan: str, implementation: list[str]) -> str:
        Stitches a plan together given implementation steps and generates a final code output using LLM.

    generate_code(task: str, plan: str, implementation: list[str], log_folder: str, code_fname: str) -> str:
        Uses the stitched plan to generate code and saves it to a file using LLM.

    call_model_openai(prompt: str, model: str = cfg.MODEL, temperature: float = 0.7) -> tuple[dict, float]:
        Queries the OpenAI LLM with a prompt and returns the result and cost.

    call_litellm(input: str, model: str = cfg.MODEL, do_cache: bool = cfg.SEED is not None, temperature: float = 0.7) -> tuple[dict, float]:
        Uses the LiteLLM library to interact with a LLM and return results and response cost.

Constants and variables:
    client: OpenAI
        An instance of the OpenAI client for model interaction.
    os.environ["AWS_REGION_NAME"]: str
        The AWS region set for the application.

Raises:
    ValueError: If the input string for LiteLLM does not contain the word JSON.
"""


from prompts import *
from utils import save_generated_code, execute_code, log_response
import json
import ast
import os
import config as cfg
from openai import OpenAI
from litellm import batch_completion


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
os.environ["AWS_REGION_NAME"] = "us-east-1"


def preformat_input(input: str | list[str]) -> list[dict[str,str]]:
	if isinstance(input, str):
		input = [input]
	return [{ "content": x, "role": "user"} for x in input]


def query_programmer(input: str, do_cache:bool=cfg.SEED is not None) -> tuple[dict, float]:
	outputs, costs = call_litellm(input, do_cache=do_cache)
	outputs = json.loads(outputs)
	return outputs, costs


def query_metric_summarizer(task: str, result: str, get_metric: bool=False) -> tuple[float, str]:
	prompt, prompt_dir = generate_metric_summarize_prompt(task, result)
	metric = call_model_openai(prompt)[0]
	metric_dir = call_model_openai(prompt_dir)[0] if get_metric else None
	return metric, metric_dir


def query_debugger(
	task: str,
	code: str,
	log_folder: str,
	code_fname: str,
	max_error_iterations: int=5,
	walk_num: int=None
) -> tuple[str, int, int, str, int]:

	error_iter = 0
	num_errors = 0
	error, return_code = execute_code(log_folder, code_fname)
	num_errors += 1 if return_code == 1 else 0
	cumulative_cost = 0
	corrected_response = code

	while return_code != 0:
		if error_iter == max_error_iterations:
			print(f"Failed to debug walk {walk_num} in {max_error_iterations} iters... increase max_error_iterations if needed.")
			save_generated_code(log_folder, "testing.py", corrected_response["Code"])
			break

		debug_prompt = generate_debugger_prompt(task, corrected_response, error)
		corrected_response, costs = query_programmer(debug_prompt)
		log_response(log_folder, f"Code ran into an issue: {corrected_response['Explanation']}\n", costs)
		save_generated_code(log_folder, "testing.py", corrected_response["Code"])  # Save corrected code
		cumulative_cost += costs
		error, return_code = execute_code(log_folder, "testing.py")
		num_errors += 1 if return_code != 0 else 0
		error_iter += 1
	
	if return_code == 0:
		corrected_response = corrected_response if isinstance(corrected_response, str) else corrected_response["Code"]
		code = corrected_response  # Keep old code
		save_generated_code(log_folder, code_fname, corrected_response)

	return code, cumulative_cost, num_errors, error, return_code


def plan_stitch(task: str, plan: str, implementation: list[str]) -> str:
	prompt = generate_plan_stitch_prompt(task, plan, implementation)
	output, cost = call_litellm(prompt)
	output = ast.literal_eval(output)
	return output["Code"], cost


def generate_code(
	task: str,
	plan: str,
	implementation: list[str],
	log_folder: str,
	code_fname: str
) -> str:
	# Stitch plan together given implementation steps
	code, cost = plan_stitch(task, plan, implementation)
	save_generated_code(log_folder, code_fname, code)
	return code, cost


def call_model_openai(prompt: str, model: str = cfg.MODEL, temperature: float = 0.7) -> tuple[dict, float]:
	if cfg.SEED is not None:
		responses = client.chat.completions.create(
			model=model, seed=cfg.SEED, messages=[{"role": "user", "content": prompt}], temperature=temperature
		)
	else:
		responses = client.chat.completions.create(
			model=model, messages=[{"role": "user", "content": prompt}], temperature=temperature
		)
	costs = 0.0  # TODO: Compute costs here
	outputs = responses.choices[0].message.content
	return outputs, costs


def call_litellm(
	input: str,
	model: str = cfg.MODEL,
	do_cache: bool = cfg.SEED is not None,
	temperature: float = 0.7
) -> tuple[dict, float]:

	messages_list = [preformat_input(input)]
	if "json" not in messages_list[0][0]['content'].lower():
		raise ValueError("We need the word JSON in the input to use JSON mode.")
	if cfg.SEED is not None:
		responses = batch_completion(
			model=model,
			messages=messages_list,
			seed=cfg.SEED,
			response_format={ "type": "json_object" },
			temperature=temperature,
			caching=do_cache
		)
	else:
		responses = batch_completion(
			model=model,
			messages=messages_list,
			response_format={ "type": "json_object" },
			temperature=temperature,
			caching=do_cache
		)
	responses = responses[0]
	costs = responses._hidden_params.get("response_cost", 0.0)
	outputs = responses.choices[0].message.content
	return outputs, costs
