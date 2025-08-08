import vertexai
import os, sys
import json
from typing import List, Dict, Any
import pandas as pd
import numpy as np

def read_json_lines(filepath: str) -> List[Dict[str, Any]]:
    """
    Reads a JSON Lines file and returns a list of Python dictionaries.
    Each line of the file is treated as a separate JSON object.

    Args:
        filepath (str): The path to the input file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries loaded from the file.
                              Returns an empty list if the file is not found
                              or an error occurs.
    """
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip any empty lines
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON on line: {line.strip()}. Error: {e}")
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except IOError as e:
        print(f"An error occurred while reading the file: {e}")
    return data

def write_json_lines(data: List[Dict[str, Any]], filepath: str) -> None:
    """
    Writes a list of Python dictionaries to a file, with each dictionary
    serialized as a JSON object on a new line. The file is encoded in UTF-8.

    This format is often referred to as 'JSON Lines' or 'ndjson'.

    Args:
        data (List[Dict[str, Any]]): The list of dictionaries to be saved.
        filepath (str): The path to the output file.
    """
    # Open the file in write mode ('w') with UTF-8 encoding.
    # The 'with' statement ensures the file is automatically closed.
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # Iterate through each dictionary in the provided list.
            for obj in data:
                # Serialize the dictionary to a JSON formatted string.
                # json.dumps() handles the encoding to a string.
                json_string = json.dumps(obj, ensure_ascii=False)
                
                # Write the JSON string to the file, followed by a newline character.
                f.write(json_string + '\n')
        print(f"Successfully wrote {len(data)} JSON objects to '{filepath}'.")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")

def print_similar_videos(query_emb: list[float], data_frame: pd.DataFrame):
    # calc dot product
    video_embs = data_frame["embedding"]
    #scores = [np.dot(eval(video_emb), query_emb) for video_emb in video_embs]
    scores = [np.dot(video_emb, query_emb) for video_emb in video_embs]
    data_frame["score"] = scores
    data_frame = data_frame.sort_values(by="score", ascending=False)

    # print results
    print(data_frame.head()[["score", "id"]])
