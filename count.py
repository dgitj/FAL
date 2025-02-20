import json

def count_nested_elements(filename):
    """Reads a JSON file and counts the total number of elements in a nested list."""
    with open(filename, "r") as file:
        nested_list = json.load(file)

    total_count = sum(len(sublist) for sublist in nested_list)
    print(f"Total number of elements: {total_count}")

# Example usage
if __name__ == "__main__":
    json_filename = "alpha0-1_cifar10_40clients.json"  # Change to your JSON file
    count_nested_elements(json_filename)
