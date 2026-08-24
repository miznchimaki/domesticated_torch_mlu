import os
import json


def validate_json_file(file_path):
    """
    Validates if a JSON file has correct syntax and is not empty.
    """
    try:
        # Attempt to open and parse the JSON file
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check if the content is empty
        # For dicts, checks if it's an empty dict {}
        # For lists, checks if it's an empty list []
        if not data:
            raise ValueError(
                f"File '{file_path}' format is correct, but content is empty."
            )

    except json.JSONDecodeError as e:
        # Catch JSON syntax errors
        raise ValueError(f"File '{file_path}' has invalid JSON format.")
    except FileNotFoundError:
        # Catch file not found error
        raise ValueError(f"File '{file_path}' does not exist.")
    except Exception as e:
        # Catch other potential errors
        raise ValueError(f"An unknown error occurred while reading the file: {e}")
