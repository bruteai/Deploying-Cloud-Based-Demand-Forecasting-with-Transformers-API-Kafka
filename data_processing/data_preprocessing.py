import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self, file_path, sequence_length=12):
        """
        :param file_path: Path to your CSV file.
        :param sequence_length: Number of time steps (months, etc.) per sequence.
        """
        self.file_path = file_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = sequence_length

    def load_data(self):
        df = pd.read_csv(self.file_path)
        df.columns = ["Month", "Sales"]
        df["Month"] = pd.to_datetime(df["Month"])
        df.set_index("Month", inplace=True)
        return df

    def preprocess(self, df):
        # Scale only the Sales column for now
        df["Sales"] = self.scaler.fit_transform(df[["Sales"]])
        return df

    def create_sequences(self, df):
        """
        Creates sequences of length `sequence_length` from the time series data.
        X will be the input sequences, y will be the next value(s) after each sequence.
        """
        data_array = df["Sales"].values
        X, y = [], []
        for i in range(len(data_array) - self.sequence_length):
            seq_X = data_array[i : i + self.sequence_length]
            seq_y = data_array[i + self.sequence_length]  # next value
            X.append(seq_X)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def run(self):
        # High-level pipeline to load, scale, and create sequences
        df = self.load_data()
        df = self.preprocess(df)
        X, y = self.create_sequences(df)
        return X, y

if __name__ == "__main__":
    processor = DataProcessor("../data/sales_data.csv", sequence_length=12)
    X, y = processor.run()

    print("Shapes:")
    print("X:", X.shape)  # (num_samples, sequence_length)
    print("y:", y.shape)  # (num_samples,)
    print("First sequence in X:", X[0])
    print("First label in y:", y[0])
