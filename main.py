import numpy as np
import pandas as pd
import configuration as config
import model as models
from keras.callbacks import ModelCheckpoint
from keras.callbacks import History
from sklearn.metrics import matthews_corrcoef

from preprocessing import Preprocessing

if __name__ == "__main__":
    # preprocess set up
    preprocessed = Preprocessing()
    paramsObj = config.Params()

    preprocessed.load_data(config.dims)

    # get train, dev and test narrays
    df = pd.read_csv(config.FILENAME)
    train = df[200:].index.tolist()
    dev = df[:100].index.tolist()
    test = df[100:200].index.tolist()

    # X, Y and train, dev, test are all narray
    model = models.DeepModel.build(params=paramsObj)

    print(model.summary())

    # save the best model & history
    filepath = "checkpoint/best_weights.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')

    history = History()

    callbacks_list = [history, checkpoint]

    model.fit(preprocessed.X[train], preprocessed.onehot_Y[train],
              # validation_data = (preprocessObj.X[dev], preprocessObj.onehot_Y[dev]),
              epochs=paramsObj.n_epoch,
              batch_size=paramsObj.batch_size,
              verbose=2,
              callbacks=callbacks_list)

    dev = np.append(dev, test)
    # predict dev set
    out = model.predict_proba(preprocessed.X[dev])
    out = np.array(out)

    # find the best threshold
    threshold = np.arange(0.1, 0.9, 0.1)
    acc = []
    accuracies = []
    best_threshold = np.zeros(out.shape[1])

    for i in range(out.shape[1]):
        y_prob = np.array(out[:, i])

        for j in threshold:
            y_pred = [1 if prob >= j else 0 for prob in y_prob]
            acc.append(matthews_corrcoef(preprocessed.onehot_Y[dev][:, i], y_pred))

        acc = np.array(acc)
        index = np.where(acc == acc.max())
        accuracies.append(acc.max())
        best_threshold[i] = threshold[index[0][0]]
        acc = []

    y_pred = np.array(
        [[1 if out[i, j] >= best_threshold[j] else 0 for j in range(preprocessed.onehot_Y[dev].shape[1])] for i in
         range(len(preprocessed.onehot_Y[dev]))])

    for idx in range(len(y_pred[0])):
        pred_col = y_pred[:, idx]
        y_col = preprocessed.onehot_Y[dev][:, idx]
        acc = np.mean(np.equal(y_col, pred_col))

        print("accuracy is {} for {} with threshold {}".format(acc, config.dims[idx], best_threshold[idx]))
