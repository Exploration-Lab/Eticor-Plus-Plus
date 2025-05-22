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

[Fill in requirements and setup instructions]

## Running Experiments

[Fill in instructions for running experiments]

## Citation

[Fill in citation information when published]

## Contributors

[Fill in contributor information]

## License

[Fill in license information]
