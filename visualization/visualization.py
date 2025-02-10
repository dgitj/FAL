import matplotlib.pyplot as plt

# Data for each labeling rate
cycles = [1, 2, 3, 4, 5, 6]

# 5% labeling rate
accuracies_5 = [28.59, 34.1, 38.44, 41.32, 42.75, 43.81]
mean_5 = 38.16833333

# 10% labeling rate
accuracies_10 = [37.76, 39.84, 45.14, 48.01, 49.72, 53.5]
mean_10 = 45.66166667

# 20% labeling rate
accuracies_20 = [46.17, 51.82, 53.87, 55.57, 58.71, 61.93]
mean_20 = 54.67833333

# Create a figure with 3 subplots side-by-side
plt.figure(figsize=(15, 4))

# Diagram for 5% labeling rate
plt.subplot(1, 3, 1)
plt.plot(cycles, accuracies_5, marker='o', color='blue', label='Accuracy')
plt.axhline(y=mean_5, color='red', linestyle='--', label=f'Mean = {mean_5:.2f}')
plt.title('5% Labeled Samples / Base=715 / Budget=357')
plt.xlabel('Cycle')
plt.ylabel('Accuracy (%)')
plt.xticks(cycles)
plt.legend()
plt.grid(True)

# Diagram for 10% labeling rate
plt.subplot(1, 3, 2)
plt.plot(cycles, accuracies_10, marker='o', color='green', label='Accuracy')
plt.axhline(y=mean_10, color='red', linestyle='--', label=f'Mean = {mean_10:.2f}')
plt.title('10% Labeled Samples / Base=1440 / Budget=720')
plt.xlabel('Cycle')
plt.ylabel('Accuracy (%)')
plt.xticks(cycles)
plt.legend()
plt.grid(True)

# Diagram for 20% labeling rate
plt.subplot(1, 3, 3)
plt.plot(cycles, accuracies_20, marker='o', color='purple', label='Accuracy')
plt.axhline(y=mean_20, color='red', linestyle='--', label=f'Mean = {mean_20:.2f}')
plt.title('20% Labeled Samples / Base=2860 / Budget=1428')
plt.xlabel('Cycle')
plt.ylabel('Accuracy (%)')
plt.xticks(cycles)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
