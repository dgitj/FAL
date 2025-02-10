import matplotlib.pyplot as plt

# Data points already separated into two lists
labeled_percent = [5, 10, 20, 35]
accuracy_percent = [43.81, 53.5, 61.93, 72.93]

# Create the plot
plt.figure(figsize=(6, 4))
plt.plot(labeled_percent, accuracy_percent, marker='o', linestyle='-', color='blue')

# Set labels and title
plt.xlabel('Final % Labeled Samples')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy vs. Labeled Samples')

# Display grid and show plot
plt.grid(True)
plt.show()
