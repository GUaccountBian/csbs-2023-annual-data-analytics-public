import pandas as pd

# Load the CSV file into a pandas dataframe
file = input("Enter the file name: ")
df = pd.read_csv(file)

filename = file.split(".")

# Calculate the percentage of null values in each column
null_ratio = df.isnull().sum() / len(df)

# Drop the columns where the null ratio exceeds a threshold
threshold = 0.9  # 90%
drop_cols = list(null_ratio[null_ratio > threshold].index)
df = df.drop(drop_cols, axis=1)

# Save the cleaned dataframe to a new CSV file
df.to_csv(filename[0] + '_cleaned.csv', index=False)