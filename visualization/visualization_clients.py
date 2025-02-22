import matplotlib.pyplot as plt
import numpy as np

# Data points with missing values replaced by None
number_clients = [10, 20, 40, 60, 80, 100]
accuracy_kafal = [72.93, 64, 64.91, 59.01, 62.06, 52]
accuracy_random = [66.5, 52, None, None, None, 50]
accuracy_entropy = [71, 59.5, None, None, None, 50.5]
accuracy_badge = [71, 57.5, None, None, None, 50.5]

# Create the plot
plt.figure(figsize=(6, 4))
plt.plot(number_clients, accuracy_kafal, marker='o', linestyle='-', label='Kafal')
plt.plot(number_clients, accuracy_random, marker='o', linestyle='-', label='Random')
plt.plot(number_clients, accuracy_entropy, marker='o', linestyle='-', label='Entropy')
plt.plot(number_clients, accuracy_badge, marker='o', linestyle='-', label='Badge')

# Set labels and title
plt.xlabel('Number of Clients')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy vs. Number of Clients (35% Labeled Samples)')
plt.legend()  # Show legend
plt.grid(True)

# Display the plot
plt.show()
