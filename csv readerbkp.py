import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('files/temp/csvfile.csv')

# Access data in the DataFrame using column names or indexing
print(type(df.head()))
