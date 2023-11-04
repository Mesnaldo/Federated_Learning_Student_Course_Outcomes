import pandas as pd

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
data = pd.read_csv(url, sep=";")

# Convert to binary classification (pass if G3 >= 10, else fail)
data["G3"] = data["G3"].apply(lambda x: 1 if x >= 10 else 0)

# Save the updated dataset to a CSV file
data.to_csv("student_binary_classification.csv", index=False)
