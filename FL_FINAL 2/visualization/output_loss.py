import matplotlib.pyplot as plt

# Your losses_centralized data
losses_centralized = [(0, 0.6931471805599453), (1, 0.5954682784783264), (2, 0.49698775149216007), (3, 0.48816243526253145), (4, 0.41561626467679197), (5, 0.4081230084214884), (6, 0.3527222447191927), (7, 0.35523048041678884), (8, 0.3156061350315555), (9, 0.3180568042648473), (10, 0.3185266606406512)]

# Extract x and y data
x = [item[0] for item in losses_centralized]
y = [item[1] for item in losses_centralized]

# Create the line plot
plt.plot(x, y, marker='o', linestyle='-')
plt.title('Centralized Loss Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)

# Display the plot
plt.show()
