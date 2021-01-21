import pandas as pd

if __name__ == "__main__":
    labels = pd.read_csv("data/train_input/train_tiles_annotated.csv")
    print(labels.head()) 
 

