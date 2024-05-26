# file_utils.py
import json

def save_json(data, filepath):
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)
