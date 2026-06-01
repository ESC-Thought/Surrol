import json

def load_json_from_file(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)  # Load the JSON data
            return data
    except FileNotFoundError:
        print(f"File {filename} not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
filename = 'env_info.txt'  # Replace with your filename
env_info = load_json_from_file(filename)
# print(env_info)

all_cases = env_info['all']
success_cases = env_info['success']
failed_cases = [case for case in all_cases if case not in success_cases]
id_list = []
for case in all_cases:
    id = case[0]
    if id not in id_list:
        id_list.append(id)
print(id_list)

for case in failed_cases:
    id = case[0]
    if id not in id_list:
        id_list.append(id)
print(id_list)

# print(failed_cases)