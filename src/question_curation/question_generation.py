# -*- coding: utf-8 -*-
"""question_generation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1l-2THC28O6xKHvQkpImEOUySwDlHFGWP
"""

import pandas as pd
import json
import random
from google.colab import files
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set a random seed for reproducibility
random.seed(42)

# Upload your CSV file
uploaded = files.upload()

df = pd.read_csv('data/medcalcqa/test_notes.csv')

# Ensure the columns are named correctly
df.columns = ['patient_id', 'evidence', 'calculator_code', 'value', 'units', 'patient', 'calculator_name', 'trunc_note', 'valid']

# Filter the DataFrame to include only rows with 'valid' column equal to 1
input_df = df[df['valid'] == 1]

calculator_counts = input_df['calculator_name'].value_counts().reset_index()
calculator_counts.columns = ['calculator_name', 'Count']

# Select the top 10 most frequent calculators
top_10_calculators = calculator_counts.head(10)

# Plotting
plt.figure(figsize=(10, 10))
sns.barplot(
    x='Count',
    y='calculator_name',
    data=top_10_calculators,
    palette='viridis'
)
plt.xlabel('Count', fontsize=16)  # Increase font size for x-axis label
plt.ylabel('Calculator', fontsize=16)  # Increase font size for y-axis label
plt.tick_params(axis='y', labelsize=16)
plt.tick_params(axis='x', labelsize=16)
plt.show()

# Group similar calculators together
calculator_groups = {
    "Stroke": [
        "ABCD2 Score",
        "National Institutes of Health Stroke Scale/Score (NIHSS)"
    ],
    "Cardiovascular Risk": [
        "Framingham Risk Score",
        "Atherosclerotic Cardiovascular Disease (ASCVD) Risk Calculator"
    ],
    "Bleeding Risk" : [
        "HAS-BLED Score for Major Bleeding Risk",
        "CHA2DS2-VASc Score for AF"
    ],
    "Liver Function" : [
        "Child-Pugh Score for Cirrhosis Mortality",
        "Model for End-Stage Liver Disease (MELD) Score"
    ],
    "Respiratory": [
        "CURB-65 Score for Pneumonia Severity",
        "PSI/PORT Score: Pneumonia Severity Index for CAP"
    ],
    "Pulmonary": [
        "PERC Rule for Pulmonary Embolism",
        "Wells' Criteria for Pulmonary Embolism"
    ],
    "Neurological" : [
        "Glasgow Coma Scale/Score (GCS)",
        "National Institutes of Health Stroke Scale/Score (NIHSS)"
    ],
    "Critical Care" : [
        "Sequential Organ Failure Assessment (SOFA) Score",
        "The Acute Physiology and Chronic Health Evaluation II (APACHE II) score"
    ],
    "Surgical Risk": [
        "Caprini Score for Venous Thromboembolism",
        "Revised Cardiac Risk Index for Pre-Operative Risk"
    ]
}

# Create a list of all unique calculators from the input_df
all_calculators = sorted(input_df['calculator_name'].unique())

# Initialize a matrix with False
matrix_size = len(all_calculators)
cannot_coexist_matrix = np.zeros((matrix_size, matrix_size), dtype=bool)

# Create a reverse mapping from calculator name to groups
calculator_to_groups = {}
for group, calculators in calculator_groups.items():
    for calculator in calculators:
        if calculator not in calculator_to_groups:
            calculator_to_groups[calculator] = []
        calculator_to_groups[calculator].append(group)

# Fill the matrix based on groupings
for i, calc1 in enumerate(all_calculators):
    for j, calc2 in enumerate(all_calculators):
        if i != j:
            groups1 = set(calculator_to_groups.get(calc1, []))
            groups2 = set(calculator_to_groups.get(calc2, []))
            if groups1 & groups2:  # Check if there's any intersection
                cannot_coexist_matrix[i, j] = True

# Create a heatmap to visualize the matrix
plt.figure(figsize=(15, 12))
ax = sns.heatmap(cannot_coexist_matrix, annot=False, cmap='coolwarm', cbar=False, xticklabels=all_calculators, yticklabels=all_calculators, linecolor='black', linewidths=0.5)
plt.title('Cannot Coexist Matrix Heatmap')
plt.xlabel('Calculators')
plt.ylabel('Calculators')
plt.xticks(rotation=90)
plt.yticks(rotation=0)

# Save the figure as a JPEG file
plt.savefig('cannot_coexist_matrix_heatmap.jpeg', format='jpeg')

# Show the plot
plt.show()

# Reset the indices of the input_df
input_df = input_df.reset_index(drop=True)

# Function to transform each row
def transform_row(row, index, all_calculators, calculator_to_groups, none_of_above_indices, cannot_coexist_matrix):
    correct_calculator = row['calculator_name']
    correct_index = all_calculators.index(correct_calculator)

    if index in none_of_above_indices:
        # "None of the above" as the correct answer
        invalid_options = [correct_calculator] + [all_calculators[i] for i, x in enumerate(cannot_coexist_matrix[correct_index]) if x]
        valid_options = [calc for calc in all_calculators if calc not in invalid_options]
        selected_calculators = random.sample(valid_options, 4)
        options_dict = {chr(65 + i): selected_calculators[i] for i in range(4)}
        options_dict['E'] = "None of the above"
        correct_answer = "E"
    else:
        # Normal case with the correct calculator in the options
        valid_options = [calc for i, calc in enumerate(all_calculators) if not cannot_coexist_matrix[correct_index, i] and calc != correct_calculator]

        if len(valid_options) < 4:
            valid_options += random.choices(valid_options, k=4 - len(valid_options))

        selected_calculators = random.sample(valid_options, 4)

        # Ensure the correct calculator is included in the first four options
        if correct_calculator not in selected_calculators:
            selected_calculators[random.randint(0, 3)] = correct_calculator

        random.shuffle(selected_calculators)  # Shuffle the first four options
        options = selected_calculators + ["None of the above"]
        options_dict = {chr(65 + i): options[i] for i in range(5)}

        # Ensure correct answer is derived from the ground truth
        correct_answer = chr(65 + options.index(correct_calculator))

    return {
        "question": "Which of the following is the correct clinical calculator to use?",
        "note": row['trunc_note'],
        "options": options_dict,
        "answer": correct_answer
    }

# Determine indices for "None of the above" correct answers
num_none_of_above = int(0.2 * len(input_df))
none_of_above_indices = random.sample(range(len(input_df)), num_none_of_above)

# Create a JSON object with zero-padded indices starting from 1
data = {str(index + 1).zfill(4): transform_row(row, index, all_calculators, calculator_to_groups, none_of_above_indices, cannot_coexist_matrix) for index, row in input_df.iterrows()}

# Convert the dictionary to a JSON object
json_data = json.dumps(data, indent=4)

# Save the JSON object to a file and download it
output_file_path = 'data/medcalcqa/med_calc_qa.json'
with open(output_file_path, 'w') as f:
    f.write(json_data)

# Download the JSON file to your local machine
files.download(output_file_path)

from collections import Counter

# Extract the answer choices
answer_choices = [item['answer'] for item in data.values()]

# Count the occurrences of each answer choice
answer_counts = Counter(answer_choices)

# Print the distributions
print("Answer Choice Distributions:")
for answer, count in answer_counts.items():
    print(f"{answer}: {count}")

# Create a DataFrame for visualization
answer_df = pd.DataFrame.from_dict(answer_counts, orient='index', columns=['count']).reset_index()
answer_df = answer_df.rename(columns={'index': 'answer'})

# Ensure the answer choices are ordered from A to E
answer_df = answer_df.sort_values(by='answer')

# Visualize the distribution using a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='answer', y='count', data=answer_df, palette='viridis')
plt.title('Distribution of Answer Choices')
plt.xlabel('Answer Choice')
plt.ylabel('Count')
plt.show()

calculator_answers = [item['options'][item['answer']] for item in data.values()]

# Count the occurrences of each calculator name
calculator_counts = Counter(calculator_answers)

print("\nCalculator Distribution:")
for calculator, count in calculator_counts.items():
    print(f"{calculator}: {count}")

# Ensure all 36 calculators are in the count with a default of 0
for calculator in all_calculators:
    if calculator not in calculator_counts:
        calculator_counts[calculator] = 0

# Convert the Counter to a list of tuples
calculator_distribution = list(calculator_counts.items())

# Create a DataFrame for plotting
df = pd.DataFrame(calculator_distribution, columns=['Calculator', 'Count'])

# Plot the distribution using seaborn
plt.figure(figsize=(12, 8))
sorted_df = df.sort_values(by='Count')
sns.barplot(x='Count', y='Calculator', data=sorted_df, palette='husl', orient='h')
plt.xlabel('Count')
plt.ylabel('Calculator')
plt.xticks(rotation=0)
plt.show()

"""### Human Performance Question Set"""

import json

df = pd.read_csv('data/medcalcqa/test_notes.csv')
df.columns = ['patient_id', 'evidence', 'calculator_code', 'value', 'units', 'patient', 'calculator_name', 'trunc_note', 'valid']
input_df = df[df['valid'] == 1]

# Load JSON data from a file
with open('data/medcalcqa/med_calc_qa.json', 'r') as file:
    json_data = json.load(file)

# Create a list of all unique calculators from the input_df
all_calculators = sorted(input_df['calculator_name'].unique())

all_calculators.append("None of the above")

import random

# Prepare DataFrames for questions and answers
question_data = []
answer_data = []

# Collect all unique questions first
unique_calculator_questions = []

# Loop through each calculator and find all unique questions
for calc in all_calculators:
    for q_id, q_data in json_data.items():
        question_text = q_data["question"]
        note_text = q_data["note"]
        options = q_data["options"]
        correct_answer = q_data["answer"]

        # Find the answer option that matches the calculator name and is the correct answer
        matching_option = None
        for option_letter, option_name in options.items():
            if option_name == calc and option_letter == correct_answer:
                matching_option = option_letter
                break

        # If a matching option is found, create a question instance
        if matching_option:
            question_instance = {
                "id": q_id,
                "Question": question_text,
                "Note": note_text,
                "A": options.get("A", ""),
                "B": options.get("B", ""),
                "C": options.get("C", ""),
                "D": options.get("D", ""),
                "E": options.get("E", ""),
                "Answer": matching_option
            }

            unique_calculator_questions.append((calc, question_instance))
            break  # Move to the next calculator name

# Shuffle the unique calculator questions to distribute them evenly
random.shuffle(unique_calculator_questions)

# Collect unique calculator questions (assuming this part has already been populated)
seen_ids = set()

# Add unique calculator questions to the final set
for calc, question_instance in unique_calculator_questions:
    if question_instance["id"] not in seen_ids:
        question_data.append(question_instance)
        seen_ids.add(question_instance["id"])

# Ensure we have at least 100 questions by sampling additional questions if needed
random.seed(42)

while len(question_data) < 100:
    # Calculate how many more questions we need
    needed_questions = 100 - len(question_data)

    # Sample remaining unique questions from json_data that haven't been added yet
    remaining_questions = [
        {
            "id": q_id,
            "Question": q_data["question"],
            "Note": q_data["note"],
            "A": q_data["options"].get("A", ""),
            "B": q_data["options"].get("B", ""),
            "C": q_data["options"].get("C", ""),
            "D": q_data["options"].get("D", ""),
            "E": q_data["options"].get("E", ""),
            "Answer": q_data["answer"]
        }
        for q_id, q_data in json_data.items()
        if q_id not in seen_ids
    ]

    # If there are no remaining unique questions, break the loop
    if not remaining_questions:
        break

    # Sample and add additional questions
    additional_questions = random.sample(remaining_questions, k=min(needed_questions, len(remaining_questions)))

    for question_instance in additional_questions:
        question_data.append(question_instance)
        seen_ids.add(question_instance["id"])

# Ensure we have exactly 100 questions, trim if necessary
question_data = question_data[:100]

# Convert to DataFrame
question_df = pd.DataFrame(question_data)
answer_df = pd.DataFrame([{"Question ID": q["id"], "Correct Answer": q["Answer"]} for q in question_data])

# Save the shuffled DataFrames to CSV
question_df.to_csv("data/medcalcqa/human_performance_questions.csv", index=False)
answer_df.to_csv("data/answers.csv", index=False)