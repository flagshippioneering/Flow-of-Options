"""
Module for managing a case database with case-based reasoning capabilities using embeddings.

This module provides a structure for storing and managing cases for a task-solving 
system. It includes functionalities for adding new cases, updating existing cases, 
and retrieving cases based on task descriptions. The module uses LLM embedding 
models to enable comparison between task descriptions, allowing for both exact 
and approximate matching based on the semantics of the description.

Classes:
    Case:
        A data class representing an individual case containing a task description, 
        high-level plan, important steps of the plan, location of the associated
        Flow-of-Options (FoO), the best metric result from simulations, and the
        metric direction preference (whether lower or higher metric preferred).

    CodeDataBase:
        Manages a database of Cases, supporting operations such as adding, updating, 
        retrieving, and converting cases to embedding representations for similarity 
        matching using an LLM embedding model.

Functions:
    _json_to_class(self):
        Converts a JSON database file into a list of Case instances.
        
    add_case(self, c_new: Case, c_old: Case, similarity: float) -> bool:
        Adds a new Case to the database or updates an existing one based on similarity.
        
    update_case(self, case_old: Case, case_new: Case):
        Updates an existing Case with new information.
        
    _create_embedding_bank(self):
        Generates an embedding representation for the task descriptions of all Cases in the database.
        
    _retrieve_embedding_case(self, task: str) -> tuple[Case, float]:
        Retrieves the most similar Case based on task description embeddings and 
        provides the similarity score.
        
    retrieve_case_given_task(self, task: str, exact_match: bool=True):
        Finds a Case in the database matching the given task description,
        using exact string matching or semantic matching via embeddings.
        
    _convert_DB_to_dict(self):
        Converts the list of Case instances into a dictionary format suitable for JSON serialization.
        
    save_into_json(self):
        Serializes and saves the current database of Cases to a JSON file.

Use Cases:
    - Efficiently manage a repository of task-solving cases with capability for semantic retrieval.
    - Retrieve past solutions based on task descriptions to optimize and adapt future task-solving strategies.
    - Maintain a consistent and up-to-date record of performance metrics and plans associated with specific tasks.

Dependencies:
    - Transformers and PyTorch for embedding generation and computation. 
    - JSON for database serialization and deserialization.
    - Utility libraries for sequential consistency such as numpy and random for seed management.
"""


import json
import os
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
import torch
import config as cfg
import random
import numpy as np


# For Reproducibility
if cfg.SEED is not None:
    random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)


@dataclass
class Case:
    task: str           # Task description
    plan: str           # High-level plan
    ranked_plan: str    # Most important steps from the plan
    foo: str            # String denoting pickled location of FoO
    best_metric: float  # Best metric obtained from traversing the FoO
    metric_dir: str     # Whether lower or higher metrics preferred


class CodeDataBase():
    def __init__(self, path_to_DB: str, replace_cases: bool=False):
        self.db_path = path_to_DB
        self.db_path_json = os.path.join(self.db_path, "database.json")
        self.DB = []
        self.replace_cases = replace_cases
        self.embedding_bank = None  # Store cases as embeddings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if os.path.exists(self.db_path_json):
            # This will be a dict of ID: {task: X, implementation: Y, metric: value, metric_dir: high/low}
            self._json_to_class() # read from json of DB and create case DB

    def _json_to_class(self):
        with open(self.db_path_json, 'r', encoding='utf-8') as file:
            DB = json.load(file)

        for id in DB.keys():
            self.DB.append(
                Case(
                    task=DB[id]['task'],
                    foo=DB[id]['foo'],
                    plan=DB[id]['plan'],
                    ranked_plan=DB[id]['ranked_plan'],
                    best_metric=DB[id]['best_metric'],
                    metric_dir=DB[id]['metric_dir']
                )
            )

    def add_case(self, c_new: Case, c_old: Case, similarity: float) -> bool:
        # Add the new case to the case repository.
        new_case = False
        if (c_old is not None) and (similarity > 0.9) and (self.replace_cases):
            print("Updating existing code in database...")
            self.update_case(c_old, c_new)
            new_case = False
        else:
            print("Adding new case to database...")
            self.DB.append(c_new)
            new_case = True

        return new_case

    def update_case(self, case_old: Case, case_new: Case):
        # Update case1 in database with case2
        case_old.task = case_new.task
        case_old.plan = case_new.plan
        case_old.ranked_plan = case_new.ranked_plan
        case_old.best_metric = case_new.best_metric
        case_old.metric_dir = case_new.metric_dir.lower()

    def _create_embedding_bank(self):
        # Convert case task descriptions into embeddings
        MODEL_NAME = cfg.EMBEDDING_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
        self.query_prompt = "Represent this query for retrieving relevant documents: "
        self.doc_prompt = "Represent this document for retrieval: "

        self.case_bank = {}
        for c in self.DB:
            self.case_bank[self.doc_prompt + c.task] = c

        # Construct Embedding Database
        x_inputs = self.tokenizer(
            list(self.case_bank.keys()),
            padding=True, 
            truncation= True,
            return_tensors='pt'
        )
            
        input_ids = x_inputs.input_ids.to(self.device)
        attention_mask = x_inputs.attention_mask.to(self.device)

        with torch.no_grad():
            x_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            x_outputs = x_outputs.last_hidden_state[:, 0]
            x_embedding = torch.nn.functional.normalize(x_outputs, p=2, dim=1)
            
        self.embedding_bank = x_embedding

    def _retrieve_embedding_case(self, task: str) -> tuple[Case, float]:
        # Retrieves closest case and corresponding similarity score
        if self.embedding_bank is None:
            self._create_embedding_bank()

        x_inputs = self.tokenizer(
            self.query_prompt + task,
            padding=True, 
            truncation= True,
            return_tensors='pt'
        )
        input_ids = x_inputs.input_ids.to(self.device)
        attention_mask = x_inputs.attention_mask.to(self.device)

        with torch.no_grad():
            x_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            x_outputs = x_outputs.last_hidden_state[:, 0]
            x_embedding = torch.nn.functional.normalize(x_outputs, p=2, dim=1)
        
        similarity = (x_embedding @ self.embedding_bank.T).squeeze()        
        _, ranking_index = torch.topk(similarity, 1)

        similarity = similarity.cpu().numpy().tolist() 
        ranking_index = ranking_index.item()
        similarity = [similarity] if not isinstance(similarity, list) else similarity

        return list(self.case_bank.values())[ranking_index], similarity[ranking_index]

    def retrieve_case_given_task(self, task: str, exact_match: bool=True):
        if not exact_match:
            if self.DB:
                # Do embedding retrieval if database non-empty
                print("Using embedding retrieval...")
                return self._retrieve_embedding_case(task)
        else:
            for c in self.DB:
                # Exact string match
                if c.task == task:
                    return c, 1.0
        return None, None

    def _convert_DB_to_dict(self):
        case_dict = {}
        for i, case in enumerate(self.DB):
            case_dict[i] = {
                "task": case.task,
                "foo": case.foo,
                "plan": case.plan,
                "ranked_plan": case.ranked_plan,
                "best_metric": case.best_metric,
                "metric_dir": case.metric_dir.lower()
            }
        return case_dict

    def save_into_json(self):
        # Save cases into a json file
        if self.DB:
            case_dict = self._convert_DB_to_dict()
            with open(self.db_path_json, 'w') as fp:
                json.dump(case_dict, fp)
