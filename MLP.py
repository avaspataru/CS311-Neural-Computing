import pandas as pd
import numpy as np
from matplotlib import pyplot


from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout
from keras import optimizers


def make_set(N):
        df = pd.DataFrame({'x1': [1, 1, 0, 0],'x2': [0, 1, 0, 1],'t': [1, 0, 0, 1]})
        max = 0
        list = []
        for j in range(0,N):
            for i in range(0,4):
                input = df.iloc[i]
                noised_x1 = input['x1'] + np.random.normal(scale=0.5)
                noised_x2 = input['x2'] + np.random.normal(scale=0.5)
                if( abs(input['x1'] - noised_x1) > max):
                    max = abs(input['x1'] - noised_x1)
                if( abs(input['x2'] - noised_x2) > max):
                    max = abs(input['x2'] - noised_x2)
                #noised_t =  input['t'] + np.random.normal(scale=0.5)

                noised_input = [noised_x1, noised_x2, input['t']]
                list.append(noised_input)
        dataset = pd.DataFrame(list,columns = ["x1","x2","t"])

        X = dataset[["x1", "x2"]]
        Y = dataset["t"]

        print(max)

        return X,Y

def make_model(N_neurons,X,Y):

    model = Sequential()
    model.add(Dense(N_neurons, input_dim=2, activation='sigmoid'))
    model.add(Dense(1, activation="sigmoid"))


    # Fit the model
    print("Fitting model...")
    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='mean_squared_error', optimizer=sgd , metrics=['mean_squared_error'])
    return model

#function plot_heatmap: adaptation from Nicola Branchini
def plot_heatmap(model, name):
    unitSquareMap = {"x1":[],"x2":[]}
    # Lower bound of the axes
    lo = 0
    # Upper bound of the axes
    hi = 1
    # Increase this for higher resolution map. E.g 5 means 5x5 output map
    ssf = 25
    # Creates dictionary of grid of float coordinates.
    for i in range(0,ssf+1):
        for j in range(0,ssf+1):
            unitSquareMap["x1"].append((float(hi-lo)/ssf)*i+lo)
            unitSquareMap["x2"].append((float(hi-lo)/ssf)*j+lo)

    unitSquareMap = pd.DataFrame(data=unitSquareMap)
    unitSquareMap = unitSquareMap[["x1","x2"]]
    unitSquareMap = unitSquareMap.values

    out2 = model.predict_proba(unitSquareMap)

    inp = unitSquareMap
    inp = inp[:,:2]

    # Empty square array
    outImg = np.empty((ssf+1,ssf+1))

    for i in range(0,(ssf+1)**2):
        row = inp[i]
        outImg[int(row[0]*ssf),int(row[1]*ssf)] = out2[i]

    outImg = outImg.T
    # This is prints in the same layout as the graph
    # print(np.flip(outImg,axis=0))

    pyplot.clf()
    img = pyplot.imshow(outImg,cmap="Greys",interpolation='none',extent = [lo,hi,hi,lo])
    pyplot.xlabel("X1")
    pyplot.ylabel("X2")
    # Flip vertically, as the plot plots from the top by default
    pyplot.gca().invert_yaxis()
    #pyplot.show()
    pyplot.savefig("output_"+name+".png")
    # plt.show()
    pyplot.close()



def make_square():
    vals = []
    for i in range (0,100):
        val = (i * 1.0) /100
        vals.append(val)
    list_pairs = []
    for i in vals:
        for j in vals:
            list_pairs.append([i,j])
    X_square = pd.DataFrame(list_pairs, columns = ["x1","x2"])
    return X_square


def main():
    print("MLP for XOR!")

    #TODO: show actual training set on graph (how much it covers)
    # show how the size of the training set influences results.

    training_N = [4,8,16] #(16,32,64) = (4*4, 4*8, 4*16)
    neurons = [2,4,8]

    X_test, Y_test = make_set(16)

    X_square = make_square()

    for i in range(0,3):
        for j in range(0,3):
            N = training_N[j]
            N_neurons = neurons[i]

            X_train,Y_train = make_set(N)
            model = make_model(N_neurons, X_train, Y_train)
            history = model.fit(X_train, Y_train, epochs=1000, batch_size=1, validation_data=(X_test, Y_test))
            pyplot.clf()
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

            # part 2 of displaying the output
            plot_heatmap(model, name)





main()
