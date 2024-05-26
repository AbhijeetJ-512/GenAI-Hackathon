# data_utils.py
def convert_keys_to_strings(data):
    content_with_str_keys = {}
    for key, value in data.items():
        str_key = f"{key[0]}_page_{key[1]}"
        content_with_str_keys[str_key] = value
    return content_with_str_keys
