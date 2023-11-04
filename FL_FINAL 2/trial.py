import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/Users/oumsai/Desktop/FL_FINAL 2/student_binary_classification.csv')
# Assuming you have a pandas DataFrame 'df' with a 'target' column
class_counts = df['school'].value_counts()
class_names = class_counts.index

plt.figure(figsize=(8, 6))
plt.bar(class_names, class_counts)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()

