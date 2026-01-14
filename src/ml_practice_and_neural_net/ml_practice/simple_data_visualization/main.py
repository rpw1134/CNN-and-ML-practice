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

    # Selecting a single value, returns a simple scalar
    selection = data_portion.at[2, "Pclass"]

    # Selecting based on position (rows 2 and 3, columns 1 and 4). Can also use slicing notation
    positional_selection = data_portion.iloc[[2,3], [1,4]]

    # NOTE: use loc when referring to indices/labels, iloc for positions (think numpy indices, rather than the actual data index)

    # Select ROWS where 'Survived' == 1
    filtered_rows = data_portion[data_portion["Survived"] == 1]

    # Filter out string cols, Select VALUES that are greater than 3. Notice the difference in using no_strings for condition, but data_portion for selection
    no_strings = data_portion.loc[:, data_portion.dtypes != object]
    selected_values = data_portion[no_strings>=1]
    print(selected_values)

def main():
    read_csv_file()

if __name__ == "__main__":
    main()