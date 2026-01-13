import pandas as pd
import numpy as np



def main():
    random_array = np.random.randn(10,5)
    random_array[0:5, 1:3] = np.nan  # Introduce NaN values for demonstration
    df = pd.DataFrame(random_array, columns=['A', 'B', 'C', 'D', 'E'])
    print(df.values)

if __name__ == "__main__":
    main()