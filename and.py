from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd
import numpy as np

def main():
    AND = {
        'X1' : [0, 0, 1, 1],
        'X2' : [0, 1, 0, 1],
        'y'  : [0, 0, 0, 1]
    }

    df = pd.DataFrame(AND)
    df

    X, Y = prepare_data(df)
    ETA = 0.3 
    EPOCHS = 10

    model = Perceptron(eta = ETA, epochs = EPOCHS)
    model.fit(X, Y)

    _ = model.total_loss()

if __name__ == '__main__':
    main()
    