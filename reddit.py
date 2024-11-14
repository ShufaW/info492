import pandas as pd
import numpy as np

reddit_csv_path = './pse_isr_reddit_comments.csv'

columns = ['self_text', 'created_time']

def keep_columns(csv_file_path, columns_to_keep):
    """Reads a CSV file and keeps only the specified columns."""
    df = pd.read_csv(csv_file_path, usecols=columns_to_keep)
    return df

reddit_cleaned_data = keep_columns(reddit_csv_path, columns)

reddit_cleaned_data.drop_duplicates(inplace=True)

reddit_cleaned_data.to_csv('df', index=False)