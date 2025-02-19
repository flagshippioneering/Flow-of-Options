# Flow-of-Options: Diversified and Improved LLM Reasoning by Thinking Through Options
Flow-of-Options (FoO) ([paper](https://arxiv.org/abs/2502.12929)) is a novel reasoning approach designed to overcome intrinsic biases in Large Language Models (LLMs) that limit the diversity of their outputs. 

## Overview

FoO is a network data structure that explicitly enumerates "options" for implementing each step in the task, as nodes in the network. Through its formulation, FoO forces the LLM to be aware of, and to explore, a broader spectrum of possibilities for completing the task, **without** any pre-training or fine-tuning.

### Flow-of-Options for AutoML

<p align="center">
<img src=assets/Summary_new.png alt="summary" width="500"/>
</p>

We incorporate FoO into an agentic system for autonomously solving Machine Learning tasks (AutoML), demonstrating its benefits on a range of tasks (including broader applicability beyond typical classification or regression tasks). FoO enforces diversity in LLM solutions through compressed, explainable representations that also support memory of past explorations when combined with case-based reasoning.

## Code
Code will be made available here soon! Thank you for your patience!

## Reference / Bibtex
```
@misc{nair2025flowofoptionsdiversifiedimprovedllm,
      title={Flow-of-Options: Diversified and Improved LLM Reasoning by Thinking Through Options}, 
      author={Lakshmi Nair and Ian Trase and Mark Kim},
      year={2025},
      eprint={2502.12929},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.12929}, 
}
```