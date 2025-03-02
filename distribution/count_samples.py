import json

# Load the JSON file
with open("alpha0-1_cifar10_10clients.json", "r") as file:
    data_splits = json.load(file)

# Count the number of samples in each sublist (each client)
client_data_sizes = {f"Client {c}": len(data_splits[c]) for c in range(len(data_splits))}

# Print the results
for client, num_samples in client_data_sizes.items():
    print(f"{client}: {num_samples} samples")
