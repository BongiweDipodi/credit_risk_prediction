# # src/data_preprocessing.py
# import pandas as pd

# def load_data(file_path):
#     """Load CSV file into a Pandas DataFrame"""
#     df = pd.read_csv(file_path)
#     return df

# def clean_data(df):
#     """Perform basic data cleaning"""

#     df = df.drop('clientid', axis=1)
    

#     df = df.dropna()
    
#     return df

# if __name__ == "__main__":
#     df = load_data("../data/original.csv")
#     df = clean_data(df)
#     print(df.head())
