import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path("./simple_data_visualization/train.csv")

def numpy_random_array():
    random_array = np.random.randn(10, 5)
    random_array[:1, 1:3] = np.nan  # Introduce NaN values for demonstration
    df = pd.DataFrame(random_array, columns=['A', 'B', 'C', 'D', 'E'])
    print(df.values)

def read_csv_file():
    df = pd.read_csv(csv_path)
    data_portion = df.head(20)

    # sorting by axis 0 sorts by index itself, by 1 sorts by column names, returns a new sorted DataFrame
    data_portion.sort_index(axis=0, ascending=False)

    # sorting by 'Survived' column in descending order. Can add more columns: hierarchical sorting
    data_portion.sort_values(by=["Survived"], ascending=False, axis=0)

    # slicing as normal (or by row)
    sliced_data = data_portion[2:10]

    # slicing based on label (or by col)
    label_sliced_data = data_portion[["Survived", "Pclass"]]

    # Slicing based on both (notice the use of .loc)
    row_and_col_slice = data_portion.loc[1:10, ["Survived", "Pclass"]]
    print(row_and_col_slice)

def main():
    read_csv_file()

if __name__ == "__main__":
    main()