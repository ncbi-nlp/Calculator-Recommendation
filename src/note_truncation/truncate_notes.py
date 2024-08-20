import json
import os
import pandas as pd
import openai
from openai import AzureOpenAI

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# Define the model and system message
model = "gpt-4-patient-truncate"

system_message = "You are a helpful medical assistant. You are to truncate patient notes based on the provided calculator name, calculator code, value, and evidence. The expected output should not contain the provided calculator information so that we can ask what calculator to use at the end of the note."

def create_user_message(row):
    return f"""Calculator Name: {row['calculator_name']}\nCalculator Code: {row['calculator_code']}\nValue: {row['value']}\nPatient Note: {row['patient']}\nEvidence: {row['evidence']}\n\nTruncated Note: """

# Read input CSV files and sample from PMC-Patients that have shown to have certain instances of specific medical calculator mentions
project_home = os.getenv('PROJECT_HOME')
input_path = os.path.join(project_home, 'data/medcalcqa/cleaned_calc_notes.csv')
input_df = pd.read_csv(input_path)

# For storing API calls that have already been completed:
output_path = 'truncated_notes.json'

if os.path.exists(output_path):
    with open(output_path, "r") as f:
        outputs = json.load(f)
else:
    outputs = {}

data = []

for index, row in input_df.iterrows():
    
    row_id = str(row["patient_id"])
    evidence = row["evidence"]

    # Check if the row_id and evidence pair already exists in the outputs
    if row_id in outputs:
        existing_evidences = [entry["evidence"] for entry in outputs[row_id]]
        if evidence in existing_evidences:
            print(f"Skipping row {index} as it already exists in the outputs.")
            continue

    # Combine the initial prompt with the text to analyze
    prompt = create_user_message(row)
    # Add user prompt to messages
    messages = [
        {"role": "system", "content": system_message},
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
    
    truncated_note = response.choices[0].message.content
    print(row_id)
    print(truncated_note)

    # Append the result to the data list for DataFrame creation
    row["trunc_note"] = truncated_note
    data.append(row)

    # Ensure the row_id key exists in the dictionary and is a list
    if row_id not in outputs:
        outputs[row_id] = []

    # Prepare the full row data to be stored in the outputs JSON
    row_data = row.to_dict()
    row_data["trunc_note"] = truncated_note

    # Append the full row data to the existing list
    outputs[row_id].append(row_data)

    # Save the updated outputs to the JSON file
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=4)    

# Convert the list to a DataFrame
output_df = pd.DataFrame(data)
# Save the DataFrame to a CSV file
output_df.to_csv('data/medcalcqa/trunc_calc_notes.csv', index=False)

