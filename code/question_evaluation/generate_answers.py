"""
Generate answers to questions from MedCalcQA
"""

import json
import os
import sys
from openai import AzureOpenAI

client = AzureOpenAI(
	api_version="2024-02-01",
	azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
	api_key=os.getenv("OPENAI_API_KEY"),
)

def clean_response(response):
	# Remove code block delimiters if present
	if response.startswith("```json") and response.endswith("```"):
		response = response[7:-3].strip()
	return response

if __name__ == "__main__":
	# Configuration
	task = sys.argv[1]
	model = sys.argv[2]
	output_path = f"medcalcqa_results/{task}_{model}.json" 
	
	if os.path.exists(output_path):
		outputs = json.load(open(output_path))
	else:
		outputs = {}

	if task == "medcalcqa":
		dataset = json.load(open("med_calc_qa.json"))
		system = 'You are a helpful medical assistant. Your task is to answer questions related to a patient note and medical calculator options. Please organize your answer in JSON dicts formatted as {"explanation": Str, "answer": "A"|"B"|"C"|"D"|"E"}.' 

	acc_list = []

	for entry_id, entry in dataset.items():
		if entry_id in outputs:
                	continue

		prompt = f"Here is the patient note: {entry['note']}\n\n{entry['question']}\n\nOptions:\n"
		for option_key, option_value in entry['options'].items():
			prompt += f"{option_key}: {option_value}\n"

		messages = [
			{"role": "system", "content": system},
			{"role": "user", "content": prompt}
		]
		
		try:
			response = client.chat.completions.create(
				model=model,
				messages=messages,
				temperature=0,
			)
		except:
			continue

		output = response.choices[0].message.content

		if model == "gpt-4o":
			output = clean_response(output)

		try:
			outputs[entry_id] = json.loads(output)
		except:
			outputs[str(entry_id)] = output

		with open(output_path, "w") as f:
			json.dump(outputs, f, indent=4)
