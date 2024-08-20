# Calculator-Recommendation

## Introduction
MedCalcQA is a medical calculator dataset used to benchmark LLMs ability to recommend clinical calculators. Each instance in the dataset consists of a truncated patient note, a question asking to recommend a specific clinical calculator, answer options (including "None of the above"), and a final answer value. Our dataset covers 35 different calculators. This dataset contains a training dataset of about 5,000 instances and a testing dataset of 1,007 instances.

## Configuration
To create one's own version of MedCalcQA, one must first set up the OpenAI API either directly through OpenAI or through Microsoft Azure. Here we use Microsoft Azure because it is compliant with Health Insurance Portability and Accountability Act (HIPAA). Please set the enviroment variables accordingly:
```python
export OPENAI_ENDPOINT=YOUR_AZURE_OPENAI_ENDPOINT_URL
export OPENAI_API_KEY=YOUR_AZURE_OPENAI_API_KEY
```
The code has been tested with Python 3.9. Please install the required Python packages by:
```python
pip install -r requirements.txt
```

## Datasets
We used data from PMC-Patients, a resource that is publicly available on HuggingFace: https://huggingface.co/datasets/zhengyun21/PMC-Patients. Please download the raw dataset by:
```python
wget https://huggingface.co/datasets/zhengyun21/PMC-Patients/resolve/main/PMC-Patients.csv
```
or
```python
from datasets import load_dataset
import pandas as pd
dataset = load_dataset("zhengyun21/PMC-Patients", split="train")
df = dataset.to_pandas()
```

## Calculator Note Extraction
The first step in curating MedCalcQA is extracting evidence of calculator usage in the PMC-Patients dataset. We used GPT-4o and a list of 35 calculators that can be found on MDCalc.com to extract calculator use instances given relevant examples. When using this extraction script, ensure that file paths are correct such that PMC-Patients (previously loaded) and the "med_calc_table.csv" (found in Calculator-Recommendation/src/note_extraction) are properly loaded.
```python
python med_calc_prompt_gpt4o.py
```

## Calculator Note Truncation
After extracting calculator instances, we truncated notes at the mention of extracted calcualtors using a fine-tuned model of GPT-4o. This fine-tuned model was provided with 72 instances of baseline notes, sentence-level evidence of calculators, and ideal truncated notes. The instances can be found in truncation_examples.csv and the OpenAI fine-tuning framework can be fed this information which is formatted with the following script:
```python
python truncation_examples.py
```

## Question Curation
We then curated questions using our ground-truth calculator evidence, truncated notes, and answer options. Question-answer pairs contained the relevant truncated note and five options, include "E. None of the above". Thus, about 1/5 of all generated questions have "E. None of the above" as the correct answer. The process of question curation can be accomplished with the following script"
```python
python question_generation.py
```

## Question evaluation
We then evaluated the ability of 8 LLMs to choose answer choices and recommend calculators using the curated questions. An example script for generating answers to these questions can be used as the following:
```python
python generate_answers.py
```

## Acknowledgements
This work was supported by the Intramural Research Programs of the National Institutes of Health, National Library of Medicine.

## Disclaimer
This tool shows the results of research conducted in the Computational Biology Branch, NCBI/NLM. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. More information about NCBI's disclaimer policy is available.
