# Calculator-Recommendation

## Introduction
MedQA-Calc is a medical calculator dataset used to benchmark LLMs ability to recommend clinical calculators. Each instance in the dataset consists of a truncated patient note, a question asking to recommend a specific clinical calculator, answer options (including "None of the above"), and a final answer value. Our dataset covers 35 different calculators. This dataset contains a training dataset of about 5,000 instances and a testing dataset of 1,007 instances.

## Configuration
To create one's own version of MedQA-Calc, one must first set up the OpenAI API either directly through OpenAI or through Microsoft Azure. Here we use Microsoft Azure because it is compliant with Health Insurance Portability and Accountability Act (HIPAA). Ensure that an appropriate PROJECT_HOME path has been set, and please set the enviroment variables accordingly:
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
The first step in curating MedQA-Calc is extracting evidence of calculator usage in the PMC-Patients dataset. We used GPT-4o and a list of 35 calculators that can be found on MDCalc.com to extract calculator use instances given relevant examples. When using this extraction script, ensure that file paths are correct such that PMC-Patients (previously loaded) and the "med_calc_table.csv" (found in Calculator-Recommendation/src/note_extraction) are properly loaded. We also performed additional mergine and post-processing after extraction.
```python
python src/note_extraction/med_calc_prompt_gpt4o.py
python src/note_extraction/merge_patient_to_calculator.py
python src/note_extraction/clean_notes.py
```

## Calculator Note Truncation
After extracting calculator instances, we truncated notes at the mention of extracted calcualtors using a fine-tuned model of GPT-4o. This fine-tuned model was provided with 72 instances of baseline notes, sentence-level evidence of calculators, and ideal truncated notes. The instances can be found in (Calculator-Recommendation/src/note_truncation/truncation_examples.csv) and the OpenAI fine-tuning framework can be fed this information which is formatted with the following script:
```python
python src/note_truncation/truncation_fine_tuning.py
```
Next, we provided our cleaned_calc_notes.csv into the fine-tuned model. Again, when using this script, ensure that file path is correct such that the "cleaned_calc_notes.csv" (found in Calculator-Recommendation/data/medcalcqa) is properly loaded.
```python
python src/note_truncation/truncate_notes.py
```
Additionally, consider using the "json_to_csv.py" script to convert JSON output to CSV format.
```python
python src/note_truncation/json_to_csv.py
```
After truncation, we then split our data into training and test sets.
```python
python src/note_truncation/split_notes.py
```

## Question Curation
We manually reviewed over 1,000 question-answer instances from our test set. When reviewing instances, we ensured that notes were properly truncated relative to calculator name, sentence evidence, and original text.

We then automatically curated questions using our ground-truth calculator evidence, truncated notes, and answer options. Question-answer pairs contained the relevant truncated note and five options, including "E. None of the above". Thus, about 1/5 of all generated questions have "E. None of the above" as the correct answer. The process of question curation can be accomplished with the following script"
```python
python src/question_curation/question_generation.py
```

## Question Evaluation
We then evaluated the ability of 8 LLMs to choose answer choices and recommend calculators using the curated questions. Here, we have provided an example script for generating answers using OpenAI models:
```python
python src/question_evaluation/generate_answers.py medcalcqa gpt-4o
```

## Acknowledgements
This work was supported by the Intramural Research Programs of the National Institutes of Health, National Library of Medicine.

## Disclaimer
This tool shows the results of research conducted in the Division of Intramural Research, NCBI/NLM. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. More information about NCBI's disclaimer policy is available.
