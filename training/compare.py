import numpy as np 


def predict_model(model, valid_gen, valid_batches, batch_size):
    """
    Predicts the network for the validation set and returns the prediction of the 
    network as well as the ground truth for the validation set.
    :param model: keras.Model = model to compare
    :param valid_gen: Iterable = validation generator 
    :param valid_batches: int = number of batches required to get all validation data
    :param batch_size: int = batch size allowed for prediction
    :return: Tupel(np.ndarray, np.ndarray)
    """
    valid_in = []
    valid_out = []
    count = 0
    for i in valid_gen():
        if count < valid_batches:
            valid_in.append(i[0])
            valid_out.append(i[1])
            count += 1
        else:
            break

    valid_in = np.concatenate(valid_in, axis=0)
    valid_out = np.concatenate(valid_out, axis=0)
    print(valid_in.shape)
    print(valid_out.shape)

    network = model.predict(valid_in, use_multiprocessing=True, batch_size=batch_size)
    print(network.shape)
    return network, valid_out


def compare(predict, truth, cutoff):
    """
    Categorise a each prediction as true positive, false positive, true negative, or false negative 
    a calculate the number of occurances. It is normed, so that the sum over the four number is 1.
    :param predict: np.ndarray = prediction of the network
    :param truth: np.ndarray = ground truth for the prediction (needs the same shape as predict)
    :param cutoff: float = cutoff value at which point a value predict in is counted as True
    :return: Tuple(int, int, int, int)
    """
    pred_bool = predict >= cutoff
    total = 1
    for i in truth.shape:
        total = total * i
    tp = np.sum(np.logical_and(pred_bool, truth))/total
    tn = np.sum(np.logical_and(~pred_bool, ~truth))/total
    fp = np.sum(np.logical_and(pred_bool, ~truth))/total
    fn = np.sum(np.logical_and(~pred_bool, truth))/total
    return tp, tn, fp, fn


def evaluate_model(model, valid_gen, valid_batches, batch_size):
    """
    predicts a model, saves the predictions, compare the predicitons to the ground truth for mulitple cutoffs
    and returns the result.
    :param model: keras.Model = model to compare
    :param valid_gen: Iterable = validation generator 
    :param valid_batches: int = number of batches required to get all validation data
    :param batch_size: int = batch size allowed for prediction
    :return: np.ndarray(dtype=float, shape=(4, int))
    """
    predict, truth = predict_model(model, valid_gen, valid_batches, batch_size)
    np.save('prediction.npy', np.concatenate((predict, truth), axis=1))
    cuts = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    quality = np.zeros((4, 19))
    for i in range(quality.shape[1]):
        quality[:, i] = compare(predict, truth, cuts[i])
    return quality

