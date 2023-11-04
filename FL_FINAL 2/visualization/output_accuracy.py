import matplotlib.pyplot as plt

# Your accuracy data
accuracy_data = {'accuracy': [(0, 0.46), (1, 1.0), (2, 0.84), (3, 0.98), (4, 0.9), (5, 0.98), (6, 0.96), (7, 0.98), (8, 0.98), (9, 0.98), (10, 0.98)]}

# Extract x and y data
accuracy = accuracy_data['accuracy']
x = [item[0] for item in accuracy]
y = [item[1] for item in accuracy]

# Create the line plot
plt.plot(x, y, marker='o', linestyle='-')
plt.title('Accuracy Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.grid(True)

# Display the plot
plt.show()
