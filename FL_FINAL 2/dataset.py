import pandas as pd
import random

# Define the number of samples
num_samples = 1000

# Generate random student data
data = {
    'study_hours': [random.uniform(0, 10) for _ in range(num_samples)],
    'attendance_percentage': [random.uniform(0, 100) for _ in range(num_samples)],
    'exam_score': [random.uniform(0, 100) for _ in range(num_samples)],
}

# Define a pass/fail threshold
pass_threshold = 75

# Create the target variable (pass or fail)
data['pass'] = [1 if score >= pass_threshold else 0 for score in data['exam_score']]

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv('student_dataset.csv', index=False)

