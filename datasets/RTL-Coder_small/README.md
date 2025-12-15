---
task_categories:
- text-generation
language:
- en
tags:
- verilog
- RTL
- EDA
- Hardware
pretty_name: 'VeriReason Verilog Filtered Dataset'
---
# RTL-Coder_small

For implementation details, visit our GitHub repository: [VeriReason](https://github.com/NellyW8/VeriReason)

Check out our paper: [VeriReason: Reinforcement Learning with Testbench Feedback for Reasoning-Enhanced Verilog Generation](https://arxiv.org/abs/2505.11849)

## Update Log
2025.05.17: Initial release of Nellyw888/Verireason-RTL-Coder_7b_reasoning_tb

## Project Description
This study introduces VeriReason, a novel approach utilizing reinforcement learning with testbench feedback to enhance the performance of pre-trained models for Verilog RTL code generation. VeriReason combines supervised fine-tuning with Guided Reward Proximal Optimization (GRPO) reinforcement learning, specifically tailored for RTL code generation. Using our curated high-quality training examples alongside a feedback-driven reward model, VeriReason achieves 83.1% functional correctness on the VerilogEval Machine benchmark, substantially outperforming both comparable-sized models and much larger commercial systems like GPT-4 Turbo.
The model integrates explicit reasoning capabilities with reinforcement learning for Verilog generation, establishing a new state-of-the-art for automated RTL synthesis. Our 7B parameter model based on Code Llama demonstrates up to a 2.8× increase in first-attempt functional correctness compared to baseline methods and exhibits robust generalization to unseen designs.

## Installation
To install this project, follow these steps:
1. Clone the repository: git clone https://github.com/NellyW8/VeriReason.git
2. Navigate to the project directory: cd VeriReason
3. Install the dependencies as specified in the repository

## Dataset Summary

RTL-Coder_small is a filtered dataset of RTLCoder, with the most effective data points for GRPO on LLM.

## Dataset Creation

### Source Data
- **Base Dataset**: Built and selected upon the RTLCoder dataset
- **Enhancements**: 
  - Added explicit reasoning steps using GPT-4.1
  - Improved code quality to better follow instructions
  - Included testbenches generated with GPT-4.1
  - Incorporated simulation results from running testbenches with the generated code

## Dataset Structure

### Data Instances
Each instance in the dataset contains:
- `id`: Unique identifier for the example
- `instruction`: Problem specification for RTL design
- `output`: Generated Verilog code solution

### Data Fields
- **instruction**: String containing the RTL design problem statement (average length: ~3,973 characters)
- **output**: String containing the Verilog code solution (average length: ~2,024 characters)

## Usage Example
To use this dataset for model training:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("Nellyw888/RTL_Coder-small")

# Example of accessing an entry
example = dataset['train'][0]
instruction = example['instruction']
reasoning_and_solution = example['output']
```

## Models Trained Using This Dataset
- [Nellyw888/VeriReason-Qwen2.5-7b-SFT-Reasoning](https://huggingface.co/Nellyw888/VeriReason-Qwen2.5-7b-SFT-Reasoning) - A 7B parameter model based on Qwen2.5-Coder-7B-Instruct, fine-tuned using this dataset with both supervised learning and reinforcement learning techniques.

- ## Citation

Please cite our paper if you use our model or dataset:

```bibtex
@misc{wang2025verireason,
      title={VeriReason: Reinforcement Learning with Testbench Feedback for Reasoning-Enhanced Verilog Generation}, 
      author={Yiting Wang and Guoheng Sun and Wanghao Ye and Gang Qu and Ang Li},
      year={2025},
      eprint={2505.11849},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.11849}, 
}
```