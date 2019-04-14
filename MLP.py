import pandas as pd
import numpy as np

#from keras.models import Sequential
#from keras.layers import Dense

def main():
    print("MLP for XOR")

    #TODO: show actual training set on graph (how much it covers)
    # show how the size of the training set influences results.

    N = 100 # 4N = size of training dataset
    df = pd.DataFrame({'x1': [1, 1, 0, 0],'x2': [0, 1, 0, 1],'t': [1, 0, 0, 1]})

    list = []
    for j in range(0,N):
        for i in range(0,4):
            input = df.iloc[i]
            noised_x1 = input['x1'] + np.random.normal(scale=0.5)
            noised_x2 = input['x2'] + np.random.normal(scale=0.5)
            noised_t =  input['t'] + np.random.normal(scale=0.5)

            noised_input = [noised_x1, noised_x2, noised_t]
            list.append(noised_input)
    dataset = pd.DataFrame(list,columns = ["x1","x2","t"])

    X = dataset[:,0:1]
    Y = dataset[:,2]

    model = Sequential()
    model.add(Dense(2, input_dim=4, activation='sigmoid'))
    model.add(Dense(8, activation='sigmoid'))
    model.add(Dense(1, activation="sigmoid"))


main()
