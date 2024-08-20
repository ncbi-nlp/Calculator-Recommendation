import csv
import json
import os
import pandas as pd
import re
from tabulate import tabulate
import openai
from openai import AzureOpenAI
import random

# Initialize the Azure OpenAI client
client = AzureOpenAI(
	api_version="2024-02-01",
	azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
	api_key=os.getenv("OPENAI_API_KEY")
)

# Define the model and system message
model = "gpt-4o"

system = 'You are a helpful assistant for extracting medical calculator values from patient notes. ' 

# List of calculators to be considered, must continue to add calculators to this table.
system += "Identify only the calculators that are found in the following table:\n"
calculator_table = pd.read_csv('med_calc_table.csv')
codes  = calculator_table['Calculator Code'].tolist()
table = tabulate(calculator_table, headers='keys', tablefmt='grid')
system += table

print(system)

system += '\nYour task is to identify and extract the exact mentions of the above calculators in the patient note. Identify the calculator names and their associated values and output this data into a complete JSON string formatted as List[Dict{"evidence": Str(the_exact_raw_sentence), "calculator_code": Str(code_of_the_calculator), "value": Str(score_of_the_calculator), "units": Str(units_of_calculation)}]. '
system += 'Use only the keys provided in the table above, and do not use empty strings for calculator_code or value variables. '
system += 'Do not create any new calculator_code names or values. Remember to ensure that the output is in JSON format.'

# Read input CSV files and sample from PMC-Patients that have shown to have certain instances of specific medical calculator mentions
project_home = os.getenv('PROJECT_HOME')
input_path = os.path.join(project_home, 'PMC_Patients', 'PMC_Patients.csv')
input_df = pd.read_csv(input_path)

# Specify number of rows to be sampled
# sampled_rows = input_df.sample(n=10000, random_state=42)
# sampled_rows.to_csv('random_10000_sample.csv', index=False)

def clean_json_string(json_string):
    # Remove unwanted characters and escape sequences
    if json_string is not None:
        cleaned_string = json_string.replace('\\n', '').replace('\\', '').replace("```json", "").replace("```", "").strip()
    else:
        cleaned_string = ''
    
    # Further cleaning could be done if necessary, e.g., removing unwanted prefixes/suffixes
    # Handle cases where the JSON content might be wrapped in other text
    if cleaned_string.startswith('"') and cleaned_string.endswith('"'):
        cleaned_string = cleaned_string[1:-1]

    return cleaned_string

def parse_json_string(json_string):
    # Clean the JSON string
    clean_output = clean_json_string(json_string)
    # Attempt to load the JSON content
    try:
        json_output = json.loads(clean_output)
    except json.JSONDecodeError as e:
        print(f"Error decoding: {e}")
        json_output = {}

    return json_output

# Function for post-processing
def filter_output(json_objects, keys_to_check, codes, allow_empty_keys=None):
    if allow_empty_keys is None:
        allow_empty_keys = []

    filtered_objects = []

    for obj in json_objects:
        if isinstance(obj, list):
            for sub_obj in obj:
                if should_include(sub_obj, keys_to_check, codes, allow_empty_keys):
                    filtered_objects.append(sub_obj)
        else:
            if should_include(obj, keys_to_check, codes, allow_empty_keys):
                filtered_objects.append(obj)

    return filtered_objects

def should_include(obj, keys_to_check, codes, allow_empty_keys):
    for key in keys_to_check:
        if key in obj:
            if obj[key] is None or obj[key] == '' or obj[key] == 'nan':
                if key not in allow_empty_keys:
                    return False
        if key == 'calculator_code' and obj.get('calculator_code') not in codes:
            return False
    return True

output_path = 'med_calc_prompt_gpt4o.json'

if os.path.exists(output_path):
    with open(output_path, "r") as f:
        outputs = json.load(f)
else:
    outputs = {}

data = []

for index, row in input_df.iterrows():
    
    row_id = str(row["patient_id"])
    note = row["patient"]

    if row_id in outputs:
        for patient_data in outputs[row_id]:
            row_data = {
                "patient_id": row_id,
            }
            for key in ['evidence', 'calculator_code', 'value', 'units']:
                if key in patient_data:
                    row_data[key] = patient_data[key]
            data.append(row_data)
        continue    

    # Combine the initial prompt with the text to analyze
    prompt = f"Here is the patient note: {note}"

    # Add user prompt to messages
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    try:
        # Generate response from the model
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
    except openai.BadRequestError as e:
        # Check if the error is due to content filtering
        if 'content_filter' in str(e):
            print(f"Content filtering triggered for row {index}. Skipping this prompt.")
        else:
            continue
    
    output = response.choices[0].message.content 

    json_output = parse_json_string(output)

    filtered_output = filter_output(json_output, ['calculator_code', 'value'], codes)
    
    # Ensure the row_id key exists in the dictionary and is a list
    if row_id not in outputs:
        outputs[row_id] = []

    # Append the new filtered_output to the existing list
    outputs[row_id].extend(filtered_output)

    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=4)

    # Prepare the row data
    for item in filtered_output:
        row_data = {
            "patient_id": row_id,
        }

        for key in ['evidence', 'calculator_code', 'value', 'units']:
            if key in item:
                row_data[key] = item[key]

        # Append the row data to the list
        data.append(row_data)

# Convert the list to a DataFrame
output_df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
output_df.to_csv('data/medcalcqa/pmc_patients_calc_gpt4o.csv', index=False)

