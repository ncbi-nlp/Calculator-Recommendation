# Author: Nick Wan
# Purpose: Merge notes and calculator information
import pandas as pd
import os

# Merge PMC_Patients notes with the GPT-4o identified calculators
project_home = os.getenv('PROJECT_HOME')
input_path = os.path.join(project_home, 'PMC_Patients', 'PMC_Patients.csv')

# Load the CSV files
# Medical calculators output
file1 = pd.read_csv('data/medcalcqa/pmc_patients_calc_gpt4o.csv')
# PMC Patients
file2 = pd.read_csv(input_path)
# Calculator full length names
file3 = pd.read_csv('src/note_extraction/med_calc_table.csv')

# Merge the files on the 'patient_id' column
merged_df = pd.merge(file1, file2[['patient_id', 'patient']], on='patient_id', how='left')
# Convert the 'value' column to numeric, forcing non-numeric values to NaN, and then drop those rows
merged_df['value'] = pd.to_numeric(merged_df['value'], errors='coerce')
merged_df = merged_df.dropna(subset=['value'])
print("Dataframe size: ", merged_df.shape)

# Drop rows in the 'evidence' column that do not contain any numerical data
merged_df = merged_df[merged_df['evidence'].str.contains(r'\d', na=False)]
print("Dataframe size: ", merged_df.shape)

# Final merge with calculator names
merged_df = pd.merge(merged_df, file3[['calculator_name', 'calculator_code']], on='calculator_code', how='left')

# Save the merged data to a new CSV file
merged_df.to_csv('data/medcalcqa/raw_calculator_usage.csv', index=False)
