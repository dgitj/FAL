import matplotlib.pyplot as plt

# Data points already separated into two lists
number_clients = [10, 20, 100]
accuracy_percent = [72.93, 64, 52]

# Create the plot
plt.figure(figsize=(6, 4))
plt.plot(number_clients, accuracy_percent, marker='o', linestyle='-', color='blue')

# Set labels and title
plt.xlabel('Number of Clients')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy vs. Number of Clients (35% Labeled Samples)')

# Display grid and show plot
plt.grid(True)
plt.show()
