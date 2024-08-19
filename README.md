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
