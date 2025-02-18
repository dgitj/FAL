import json
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

alpha = 0.1
num_partitions = 40

partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label",
                                   alpha=alpha, min_partition_size=10,
                                   self_balancing=True)
fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})


# Generate sample indices based on enumeration
global_indices = list(range(50000))  # CIFAR-10 has 50,000 training samples

# Extract only indices from partitions
partition_data = {
    f"partition_{partition_id}": [
        global_indices[i] for i, sample in enumerate(fds.load_partition(partition_id))
    ]
    for partition_id in range(num_partitions)
}



# Save data to JSON
filename = f"alpha{alpha}_{num_partitions}clients.json"
with open(filename, "w") as json_file:
    json.dump(partition_data, json_file, indent=4)

print(f"Partition sizes saved to {filename} with alpha={alpha} and {num_partitions} clients.")