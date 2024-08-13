import json
import pandas as pd

def json_to_csv(json_file_path, csv_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Prepare a list to hold all the records
    records = []

    # Iterate through the JSON data to create records
    for patient_id, details in data.items():
        for record in details:
            # Add each record to the list
            records.append(record)
    
    # Convert the list of records to a pandas DataFrame
    df = pd.DataFrame(records)
    
    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)

# Define the file paths
json_file_path = 'truncated_notes.json'
csv_file_path = 'final_refined_notes.csv' 

# Call the function to convert JSON to CSV
json_to_csv(json_file_path, csv_file_path)
