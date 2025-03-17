"""
Prompts for generating task plans, evaluating steps, and debugging code using large language models.

This module provides utility functions to generate various prompts that assist in task planning, 
execution, and analysis by leveraging advanced language models. The generated prompts facilitate 
operations like step-by-step task decomposition, ranking task steps by impact, generating 
implementation options, and performing debugging, among others.

Functions:
    generate_planner_prompt(task: str) -> str:
        Generates a prompt for creating a detailed step-by-step plan for a given task, excluding specific non-essential steps.
    
    generate_rank_steps_prompt(task: str, plan: str) -> str:
        Constructs a prompt to rank steps from a high-level plan based on their relevance to task accuracy.

    generate_nodes_prompt(task: str, step: str, num_options: int, prev_steps: list[str]) -> str:
        Creates a prompt for generating diverse implementation options for a specific plan step while considering previous steps.

    generate_nodes_given_case_prompt(task: str, step: str, num_options: int, prev_steps: list[str], case_options: list[str]) -> str:
        Generates a prompt for creating diverse implementation options for a plan step while excluding specific cases and considering previous steps.

    generate_plan_stitch_prompt(task: str, plan: str, implementation: list[str]) -> str:
        Produces a prompt that guides the creation of complete Python code from a task plan and specific implementation details.

    generate_debugger_prompt(task: str, code: str, error: str) -> str:
        Constructs a prompt for debugging Python code by identifying and explaining errors and proposing fixes.

    generate_metric_summarize_prompt(task: str, result: str) -> tuple[str, str]:
        Creates prompts for extracting key metrics from execution traces and determining if the task is a classification or regression.

    generate_check_consistency_prompt(plan: str, step: str) -> str:
        Generates a prompt for verifying the consistency of a new plan step with existing steps, identifying contradictions.

    generate_adapter_prompt(task: str, plan: list[str]=None, options: list[str]=None) -> str:
        Produces a prompt for adapting a plan or options to a new task, making necessary adjustments specific to the task context.

Use Cases:
    - Creating detailed plans for machine learning tasks.
    - Ranking task steps by impact on task performance.
    - Generating diverse, task-specific implementation strategies.
    - Debugging and refining Python code for task execution.
    - Adapting existing plans and options to new task requirements.
"""


def generate_planner_prompt(task: str) -> str:
    prompt = f"""
    You are an expert ML scientist provided with the following task:
    task: {task}

    Think through the task step-by-step and produce a highly detailed list of steps involved in solving this task.
    Do not include steps for visualization, logging, hyperparameter tuning, and documentation.
    Do not include any code in your response.\n
    """

    structure = """
    Generate your output in the following JSON-friendly format:
    {
    "Step1": <insert a string denoting the step #1>,
    "Step2": <insert a string denoting the step #2>,
    ...
    }
    """
    return prompt + structure


def generate_rank_steps_prompt(task: str, plan: str):
    prompt = f"""
    You are an expert ML scientist provided with a task and a high-level plan for solving the task.
    task: {task}
    high-level plan: {plan}

    Your goal is to rank the steps in the high-level plan based on their relevance to accuracy on the task.
    First, you must assess each step in the plan and identify whether it can impact accuracy on the task.
    Second, for all the steps that can impact accuracy on the task, you must rank the steps from most impactful to least impactful.
    
    Next, act as an impartial judge and validate your response by returning to the provided information and verifying that the response is correct.
    If the response is incorrect, modify the response to the correct one.
    """

    structure = """
    Generate your output in the following JSON-friendly format:
    {
    "<insert step from the plan>": <insert a number denoting the rank of the step with 1 being most impactful and -1 denoting not impactful>,
    "<insert step from the plan>": <insert a number denoting the rank of the step with 1 being most impactful and -1 denoting not impactful>,
    ...
    }
    """
    return prompt + structure


def generate_nodes_prompt(task: str, step: str, num_options: int, prev_steps: list[str]) -> str:
    s = "".join(prev_steps)

    if prev_steps:
        prompt = f"""
        You are an expert ML scientist provided with a task, and one step from a high-level plan for solving this task. You are also given implementation details for all of the steps prior to the current step.
        task: {task}
        prior steps: {s}
        current step: {step}

        Your goal is to generate diverse options for implementing the current step if it impacts final accuracy.
        Firstly, you must assess whether the implementation of the current step can impact final accuracy on the task.
        If it is expected to impact final accuracy, you must:
        First, think as broadly as you can to generate {num_options} diverse and distinct options for implementing the current step, while taking into account the prior steps.
        Second, you must verify that each of your generated options do not conflict with any of the prior steps.
        Third, you must regenerate alternatives for any options that conflict with prior steps.

        Please follow these instructions in generating your response:
        1. You must maximize the diversity across all your generated options.
        2. Your choices must be very specific so that a programmer can implement it. For e.g., instead of specifying "Use features", you must specify "Use features X, Y, Z."
        3. Your choices should not modify any of the prior steps.
        4. Do not repeat any of the prior steps.
        5. Avoid time-consuming choices such as hyperparameter tuning.
        6. Do not include code in your response.\n"""
    else:
        prompt = f"""
        You are an expert ML scientist provided with a task, and one step from a high-level plan for solving this task.
        task: {task}
        current step: {step}

        Your goal is to generate diverse options for implementing the current step if it impacts final accuracy.
        Firstly, you must assess whether the implementation of the current step can impact final accuracy on the task.
        If it is expected to impact final accuracy, you must:
        Think as broadly as you can to generate {num_options} diverse and distinct choices for implementing the current step.

        Please follow these instructions in generating your response:
        1. You must maximize the diversity across all your generated options.
        2. Your choices must be very specific so that a programmer can implement it. For e.g., instead of specifying "Use features", you must specify "Use features X, Y, Z."
        3. Avoid time-consuming choices such as hyperparameter tuning.
        4. Do not include code in your response.\n"""

    structure = """
    Generate your output in the following JSON-friendly format:
    {
        "choice 1": <insert a string denoting choice #1>,
        "choice 2": <insert a string denoting choice #2>,
        ...
    }

    Make sure to begin and terminate the strings with double quotes.
    Do not include ```python in your response.

    If the current step implementation is irrelevant to final accuracy on the task, return the current step as is.
    {
        "choice 1": <insert a string denoting the current step>
    }
    """
    return prompt + structure


def generate_nodes_given_case_prompt(task: str, step: str, num_options: int, prev_steps: list[str], case_options: list[str]) -> str:
    s = "".join(prev_steps)
    c = [f"{i + 1}. {case_options[i]}\n" for i in range(len(case_options))]
    c = "".join(c)

    if prev_steps:
        prompt = f"""
        You are an expert ML scientist provided with a task, and one step from a high-level plan for solving this task. You are also given implementation details for all of the steps prior to the current step.
        task: {task}
        prior steps: {s}
        current step: {step}

        Your goal is to generate diverse options for implementing the current step if it impacts final accuracy.
        However, you must exclude a given set of options from your generations.

        options to exclude: {c}

        Firstly, you must assess whether the implementation of the current step can impact final accuracy on the task.
        If it is expected to impact final accuracy, you must:
        First, think as broadly as you can to generate {num_options} diverse and distinct options for implementing the current step, while taking into account the prior steps.
        Second, verify that your options do not include the options that must be excluded. Regenerate alternatives otherwise.
        Third, you must verify that each of your generated options do not conflict with any of the prior steps. Regenerate alternatives otherwise.

        Please follow these instructions in generating your response:
        1. You must maximize the diversity across all your generated options.
        2. Your choices must be very specific so that a programmer can implement it. For e.g., instead of specifying "Use features", you must specify "Use features X, Y, Z."
        3. Your choices should not modify any of the prior steps.
        4. Do not repeat any of the prior steps.
        5. Avoid time-consuming choices such as hyperparameter tuning.
        6. Do not include code in your response.\n"""
    else:
        prompt = f"""
        You are an expert ML scientist provided with a task, and one step from a high-level plan for solving this task.
        task: {task}
        current step: {step}
        
        Your goal is to generate diverse options for implementing the current step if it impacts final accuracy.      
        However, you must exclude a given set of options from your generations.

        options to exclude: {c}

        Firstly, you must assess whether the implementation of the current step can impact final accuracy on the task.
        If it is expected to impact final accuracy, you must:
        First, think as broadly as you can to generate {num_options} diverse and distinct choices for implementing the current step.
        Second, verify that your options do not include the options that must be excluded. Regenerate alternatives otherwise.

        Please follow these instructions in generating your response:
        1. You must maximize the diversity across all your generated options.
        2. Your choices must be very specific so that a programmer can implement it. For e.g., instead of specifying "Use features", you must specify "Use features X, Y, Z."
        3. Avoid time-consuming choices such as hyperparameter tuning.
        4. Do not include code in your response.\n"""

    structure = """
    Generate your output in the following JSON-friendly format:
    {
        "choice 1": <insert a string denoting choice #1>,
        "choice 2": <insert a string denoting choice #2>,
        ...
    }

    Make sure to begin and terminate the strings with double quotes.
    Do not include ```python in your response.

    If the current step implementation is irrelevant to final accuracy on the task, return the current step as is.
    {
        "choice 1": <insert a string denoting the current step>
    }
    """
    return prompt + structure


def generate_plan_stitch_prompt(task: str, plan: str, implementation: list[str]) -> str:
    s = "".join(implementation)

    prompt = f"""
    You are an expert AI programmer provided with a task and a plan of steps for accomplishing the task:
    task: {task}
    plan: {plan}

    You are also given specific implementation details for some of the steps in the plan.
    {s}

    Your goal is to stitch the plan together to create a complete Python code.
    You must ensure to incorporate the specific implementation details into your final code.
    You must follow the template provided in the task description as closely as possible and integrate the implementation details into the provided template.
    
    You must not make any functional changes to the provided implementation details themselves.
    Your code must be complete and should not leave any additional steps for the user nor raise any errors.\n
    """

    structure = """
    Generate your output in the following JSON-friendly format:
    {
        "Code": <insert a string denoting your stitched code>
    }

    Make sure to begin and terminate the strings with double quotes.
    Do not include ```python in your response.
    """
    return prompt + structure


def generate_debugger_prompt(task: str, code: str, error: str) -> str:
    prompt = f"""
    You are provided with a code and an error message from running the code. Please debug the code systematically to fix the error.
    For additional context, you are also provided with the original user prompt indicating the task that the code is trying to achieve.
    
    Original prompt: {task},
    Error message: {error},
    Code: {code}

    The error may also be syntactical in which case you must fix the syntax appropriately.

    """

    structure = """
    Generate your output in the following JSON-friendly format:
    {
    "Explanation": <insert a string explaining the error and its corresponding fix>,
    "Code": <insert a string denoting the corrected Python code>
    }
    Your code must successfully complete the given task and should not raise any errors.\n
    """

    # Pass to programmer
    return prompt + structure


def generate_metric_summarize_prompt(task: str, result: str) -> str:
    in_context_example = """
    execution trace example 1:
    ```
    <some output strings>
    Final <some metric> result: 0.64
    ```
    expected output: 0.64

    execution trace example 2:
    ```
    <some output strings>
    {'caco2-wang': 0.48, 0.0021}
    ```
    expected output: 0.48
    """

    prompt_metric = f"""
    You are provided with an execution trace of having run Python code that generates an output metric at the very end.
    Please respond with the first of the final numbers in the execution trace and nothing else.

    Examples: 
    {in_context_example}
        
    Pay attention to exclude any other validation metric outcomes from your response.
        
    Execution Trace:
    {result}

    Note that your output must only consist of a float value.

    Next, act as an impartial judge and validate your response by returning to the provided information and verifying that the response is correct.
    If the response is incorrect, modify the response to the correct one.
    """

    prompt_metric_dir = f"""
    You are provided with a task.
    Firstly, you must analyze whether the provided task is a classification task, or a regression task.
    If the task is a classification task, return the string "higher" in your response.
    If the task is a regression task, return the string "lower" in your response.

    Task: {task}
    
    Your output must only consist of one word (either "higher" or "lower") and nothing else.

    Next, act as an impartial judge and validate your response by returning to the provided information and verifying that the response is correct.
    If the response is incorrect, modify the response to the correct one.
    """

    return prompt_metric, prompt_metric_dir


def generate_check_consistency_prompt(plan: str, step: str) -> str:
    prompt = f"""
    You are given a plan of steps for solving a task, and a new step.
    Your goal is to verify if the information contained in the new step contradicts any of previous steps of the plan.
    
    plan:
    {plan}

    new step:
    {step}

    The following are examples of a contradiction: 
    1. If the new step references a different model than the previous steps in the plan.
    2. If the new step references a different feature than the previous steps in the plan.

    First, assess whether the new step contradicts previous steps in the plan as described above.
    If the new step contradicts other steps in the plan, return the word: yes
    If the new step does not contradict other steps in the plan, return the word: no\n"""

    structure = """
    You must return your output in the following JSON-friendly format:
    {
    "contradiction?": <return a string denoting either yes or no depending on your assessment.>,
    "rationale": <insert a string denoting the rationale for your assessment.>
    }
    """
    return prompt + structure


def generate_adapter_prompt(task: str, plan: list[str]=None, options: list[str]=None) -> str:
    assert plan is not None or options is not None, "Please specify either plan or option."
    if plan is not None:
        plan = [f"{i + 1}. {plan[i]}\n" for i in range(len(plan))]
        prompt = f"""
        You are provided with a task. You are also provided with a high-level plan for solving a different, but related task.
        Your goal is to adapt the steps in the plan for the new task.

        Task: 
        {task}

        Plan:
        {plan}

        For each step in the plan, you must identify if that step needs to be adapted for the current task.
        If adaptation is necessary, modify the step to adapt it to the new task.
        If no adaptation is necessary, leave the step as is.
        Please do not add or remove any steps.
        Return the adapted plan.
        """

        structure = """
        Generate your output in the following JSON-friendly format:
        {
        "Step1": <insert a string denoting the step #1>,
        "Step2": <insert a string denoting the step #2>,
        ...
        }
        """

        return prompt + structure
    else:
        options = [f"{i + 1}. {options[i]}\n" for i in range(len(options))]
        prompt = f"""
        You are provided with a task. You are also provided with implementation details for solving a different, but related task.
        Your goal is to adapt the implementation details for the new task.

        Task: 
        {task}

        Implementation:
        {options}

        For each implementation, you must identify if its details need to be adapted for the current task.
        If adaptation is necessary, modify the implementation details to adapt it to the new task.
        If no adaptation is necessary, leave the implementation details as is.
        Please do not add or remove any implementations.
        Return the adapted implementations as a list.
        """

        structure = """
        Generate your output in the following JSON-friendly format:
        {
        "1": <insert a string denoting implementation #1 without the bullet point numbering>,
        "2": <insert a string denoting implementation #2 without the bullet point numbering>,
        ...
        }
        """
        
        return prompt + structure
