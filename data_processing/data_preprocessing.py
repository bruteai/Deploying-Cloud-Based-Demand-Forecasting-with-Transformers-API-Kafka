import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def load_data(self):
        df = pd.read_csv(self.file_path)
        df.columns = ["Month", "Sales"]
        df["Month"] = pd.to_datetime(df["Month"])
        df.set_index("Month", inplace=True)
        return df
    
    def preprocess(self, df):
        df["Sales"] = self.scaler.fit_transform(df[["Sales"]])
        return df

if __name__ == "__main__":
    processor = DataProcessor("../data/sales_data.csv")
    raw_data = processor.load_data()
    processed_data = processor.preprocess(raw_data)
    print(processed_data.head())
