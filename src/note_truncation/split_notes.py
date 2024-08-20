import pandas as pd
import numpy as np

# Load the CSV file into a pandas dataframe
file_path = 'data/medcalcqa/raw_calc_notes.csv'
df = pd.read_csv(file_path)

# Function to sample N instances for each unique calculator_code
def sample_instances(df, N, random_state=42):
    sampled_df = df.groupby('calculator_code').apply(lambda x: x.sample(n=min(len(x), N), replace=False, random_state=random_state)).reset_index(drop=True)
    return sampled_df

# Set the sample size and random state
N = 50
random_state = 42

# Sample instances
sampled_df = sample_instances(df, N, random_state)

# Create the training dataframe containing all other rows
training_df = df[~df.index.isin(sampled_df.index)]

# Print the number of instances for each calculator_code and the sum of all instances in sampled_df
instance_counts = sampled_df['calculator_code'].value_counts()
print(f"Number of instances for each calculator_code when N={N}:")
print(instance_counts)
print(f"Sum of all instances when N={N}: {instance_counts.sum()}")

# Save the sampled dataframe to a new CSV file
sampled_df.to_csv('data/medcalcqa/raw_notes_test.csv', index=False)

# Save the training dataframe to a new CSV file
training_df.to_csv('data/medcalcqa/raw_notes_training.csv', index=False)
