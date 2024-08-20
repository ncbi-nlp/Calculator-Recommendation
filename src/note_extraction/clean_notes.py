import pandas as pd
import random

# Load the CSV file into a DataFrame
df = pd.read_csv("data/medcalcqa/raw_calc_notes.csv")

# Remove rows with specific calculator codes
codes_to_remove = ['free_water', 'bishop', 'bwps', 'etoh']
df_filtered = df[~df['calculator_code'].isin(codes_to_remove)]

# Function to sample up to 1000 instances per calculator code
def sample_calculator_code(group):
    return group.sample(min(len(group), 1000), random_state=1)

# Apply the sampling function to each calculator code group
df_sampled = df_filtered.groupby('calculator_code').apply(sample_calculator_code).reset_index(drop=True)

# Save the final DataFrame to a CSV file
df_sampled.to_csv("data/medcalcqa/cleaned_calc_notes.csv", index=False)
