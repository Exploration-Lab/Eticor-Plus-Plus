# Eticor-Plus-Plus

A comprehensive framework for evaluating cultural etiquette understanding and biases in Large Language Models (LLMs).

## Project Overview

Eticor-Plus-Plus is a multicultural dataset to evaluate how large language models (LLMs) understand and respond to cultural etiquettes across different regions of the world. The project assesses three key dimensions:

1. **Etiquette Sensitivity Task**: How well models understand appropriate vs. inappropriate cultural behaviors. Use classification *accuracy* and *f1-score* as metrics for comparison.
2. **Etiquette Generation Task**: How consistent are the model's region-specific generated etiquette content and what words do they use to do so. Defined *Generation Alignment Score (GAS)* to measure consistency and used *Odds Ratio* for qualitative assesment of generated etiquettes.
3. **Region Identification Task**: How models attribute etiquette statements to specific regions. Defined three novel metrics (*Preference Score (PS), Bias For Score (BFS) and Bias Score Pairwise (BSP)*) to measure the biased behavior of LLMs against low-resource languages.

## Dataset

The core dataset `Eticor_plus_plus` contains cultural etiquette statements across five global regions:

- East Asia (EA)
- Middle East & Africa (MEA)
- India (INDIA)
- Latin America (LA)
- Northern Europe/Western Nations (NE)

Each statement is categorized by:

- Context group (visits, business, dining, travel)
- Region of origin
- Label (positive/negative - indicating appropriate/inappropriate behavior)

## Models Evaluated

The framework tests multiple state-of-the-art LLMs:

- ChatGPT-4o
- Gemini-1.5-flash
- Gemma-2-9B-instruct
- LLaMA-3.1-8B-instruct
- Phi-3.5-Mini-instruct

## Repository Structure

```
Eticor-Plus-Plus/
├── analysis_scripts/         # Analysis code for experimental results
│   ├── e_sensitivity/        # Etiquette sensitivity analysis
│   ├── gen_bias/             # Generation bias analysis
│   └── pref_bias/            # Preference bias analysis
├── experiment_scripts/       # Scripts for running experiments on different models
├── final_data_grouped/       # Processed final data to use for experiments
├── links/                    # Countrywise new source links for cultural etiquettes
└── notebooks/                # Jupyter notebooks for data collection and processing
```

## Installation and Setup

1. Basic libraries and modules are required such as `pytorch` with cuda enabled, `transformers`, `google-generativeai`, `goose3`, `textstat`, `glob2`, `numpy`, `pandas` etc.
2. Please take care to provide access tokens/ api access keys wherever required. For example to access gated models or paid APIs.
3. Download the data from [huggingface](https://huggingface.co/datasets/Exploration-Lab/Eticor-Plus-Plus) to use in experiments.

## Running Experiments

* For E-sensitivity task, Region Identification Task and Etiquette Generation Task, one can simply run the model-specific scripts by providing required arguments and generate multiple responses to analyse them. Directories with names containing the response numbers will be generated.
* Some tasks require running of additional scripts before the analysis can be done. For Etiquette Generation Task, it is required to run the `gen_response_dict.py` and also the `new_gen_bias_script.py` scripts before moving towards the analysis.
* Support for continuation is also available. For ChatGPT and Gemini, the script will continue getting the response from where it left (if you want to understand more, see `chatgpt_*.py` files and `gemini_*.py` files.

## Citation

[Fill in citation information when published]

## Contributors

[Fill in contributor information]

## License

[Fill in license information]
