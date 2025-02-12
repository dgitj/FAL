import json
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

alpha = 0.1
num_partitions = 40

partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label",
                                   alpha=alpha, min_partition_size=10,
                                   self_balancing=True)
fds = FederatedDataset(dataset="mnist", partitioners={"train": partitioner})
partition = fds.load_partition(0)
print(partition[0])  # Print the first example


partition_sizes = {
    f"partition_{partition_id}": len(list(fds.load_partition(partition_id)))  # Convert set to list
    for partition_id in range(num_partitions)
}
# print(sorted(partition_sizes))

# Save data to JSON
filename = f"alpha{alpha}_{num_partitions}clients.json"
with open(filename, "w") as json_file:
    json.dump(partition_sizes, json_file, indent=4)

print(f"Partition sizes saved to {filename} with alpha={alpha} and {num_partitions} clients.")