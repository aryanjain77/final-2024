import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('train.csv')

# Remove the last row
df = df[:-1]

# Save the DataFrame back to the CSV file
df.to_csv('train.csv', index=False)