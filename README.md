# Eticor-Plus-Plus[World_map.pdf](World_map.pdf)


Large Language Models (LLMs) are increasingly being used in global applications, yet their understanding of cultural norms, especially etiquettes that vary from region to region, remains underexplored. Etiquettes are not just social niceties; they are a key part of cultural identity, and making LLMs aware of them is crucial for building respectful and culturally aware AI.

To address this gap, we present EtiCor++, a curated dataset of etiquettes from around the world. This resource is designed to evaluate LLMs on their knowledge of regional etiquettes and to analyze cultural biases in their responses. We provide a suite of tasks and introduce general and novel metrics for measuring how fair and consistent LLMs are across different cultures.

Our experiments reveal that popular LLMs often show unintended regional bias, highlighting the importance of this work for all kinds of practitioners aiming to build more inclusive AI systems.

## Project Overview

Eticor-Plus-Plus is a multicultural dataset to evaluate how large language models (LLMs) understand and respond to cultural etiquettes across different regions of the world. The project assesses three key dimensions:

1. **Etiquette Sensitivity Task**: How well models understand appropriate vs. inappropriate cultural behaviors. Use classification *accuracy* and *f1-score* as metrics for comparison.
2. **Etiquette Generation Task**: How consistent are the model's region-specific generated etiquette content and what words do they use to do so. Defined *Generation Alignment Score (GAS)* to measure consistency and used *Odds Ratio* for qualitative assesment of generated etiquettes.
3. **Region Identification Task**: How models attribute etiquette statements to specific regions. Defined three novel metrics (*Preference Score (PS), Bias For Score (BFS) and Bias Score Pairwise (BSP)*) to measure the biased behavior of LLMs against low-resource languages.

## Dataset

The core dataset `Eticor_plus_plus` can be downloaded from [huggingface](https://huggingface.co/datasets/Exploration-Lab/Eticor-Plus-Plus) to use in experiments.It contains cultural etiquette statements across five global regions:

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
3. Make sure to use proper format of dataset and use correct file paths for experimentation as well as the analysis.

## Running Experiments

* For E-sensitivity task, Region Identification Task and Etiquette Generation Task, one can simply run the model-specific scripts by providing required arguments and generate multiple responses to analyse them. Directories with names containing the response numbers will be generated.
* Some tasks require running of additional scripts before the analysis can be done. For Etiquette Generation Task, it is required to run the `gen_response_dict.py` and also the `new_gen_bias_script.py` scripts before moving towards the analysis.
* Support for continuation is also available. For ChatGPT and Gemini, the script will continue getting the response from where it left (if you want to understand more, see `chatgpt_*.py` files and `gemini_*.py` files.

## Citation

```
@inproceedings{dwivedi-etal-2025-eticor-plus-plus,
    title = "{EtiCor++}: Towards Understanding Etiquettical Bias in LLMs",
    author = "Dwivedi, Ashutosh and
      Singh, Siddhant Shivdutt and
      Modi, Ashutosh",
    booktitle = "Proceedings of the 2025 Findings of Conference on Association of Computational Linguistics",
    month = dec,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    abstract = "In recent years, researchers have started analyzing the cultural sensitivity of LLMs. In this respect, Etiquettes have been an active area of research. Etiquettes are region-specific and are an essential part of the culture of a region; hence, it is imperative to make LLMs sensitive to etiquettes. However, there needs to be more resources in evaluating LLMs for their understanding and bias with regard to etiquettes. In this resource paper, we introduce EtiCor++, a corpus of etiquettes worldwide. We introduce different tasks for evaluating LLMs for knowledge about etiquettes across various regions. Further, we introduce various metrics for measuring bias in LLMs. Extensive experimentation with LLMs shows inherent bias towards certain regions."
}
```

## License

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)


The Eticor++ dataset follows [CC-BY-NC](CC-BY-NC) license. Thus, users can share and adapt our dataset if they give credit to us and do not use our dataset for any commercial purposes.
