import pandas as pd
import numpy as np
from matplotlib import pyplot


from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout
from keras import optimizers


def make_set(N):
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

        X = dataset[["x1", "x2"]]
        Y = dataset["t"]

        return X,Y

def make_model(N_neurons,X,Y):

    model = Sequential()
    model.add(Dense(N_neurons, input_dim=2, activation='sigmoid'))
    model.add(Dense(1, activation="sigmoid"))


    # Fit the model
    print("Fitting model...")
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    return model

def main():
    print("MLP for XOR")

    #TODO: show actual training set on graph (how much it covers)
    # show how the size of the training set influences results.

    training_N = [4,8,16] #(16,32,64) = (4*4, 4*8, 4*16)
    neurons = [2,4,8]

    X_test, Y_test = make_set(64)

    for i in range(0,3):
        for j in range(0,3):
            N = training_N[i]
            N_neurons = neurons[i]

            X_train,Y_train = make_set(N)
            model = make_model(N_neurons, X_train, Y_train)
            history = model.fit(X_train, Y_train, epochs=1000, batch_size=1, validation_data=(X_test, Y_test))
            pyplot.plot(history.history['mean_squared_error'])
            pyplot.plot(history.history['val_mean_squared_error'])


            #pyplot.show()
            # N_neurons gives the model and training set size the experiment
            name = "neurons_" + str(N_neurons) + "setsize_" + str(N)
            pyplot.savefig("graph_"+name+".png")

            f = open("score_"+name+".txt", "a")
            scores = model.evaluate(X_train, Y_train)
            f.write("\n on train %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            scores = model.evaluate(X_test, Y_test)
            f.write("\n on test %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            f.close()

            print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            #err = er



main()
