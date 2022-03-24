import numpy as np 

def build_generators(data, truth, split, seed, batch_size, t_len):
    """
    Builds 3 generators from the data. These generators, build the training data in real time,
    to reduce the amount of memory required. 
    These generators are:
        - train_gen: Generates training data
        - truth_gen: Generates validation data
        - comp_gen: Generates validation data in a given order (order is the array returned with the gens)

    :param data: np.ndarray[float, shape=(int, int, int, int)] = Training data set, in the form [time steps, baselines, channels, pol + (amp+phases)]
    :param truth: np.ndarray[float, shape=(int, int, int, int)] = Flag table to be used as ground truth
    :param split: float = between 0 and 1, equivivalent to tfs validation split.
    :param seed: int = seed for the np RNG, to ensure repeatablity
    :param batch_size: int = sets the batch size of the generator. Set as high as possible, but as low as required for your machine
    :param t_len: int = sets how many time steps are given to the network to predict the flag of the last
    :return: Tuple[Iterable, Iterable, Iterable, np.ndarray] = train_gen, truth_gen, comp_gen, truth_indexes
    """

    # set the RNG
    np.random.seed(seed=seed)
    # calculate cardinality of the training set
    amount = len(data)-t_len

    # draw which time step belonges to the validation set (number)
    truth_indexes = np.random.choice(amount, size=int(amount*split), replace=False) 

    # make a boolean mask for the validation set
    temp_array = np.zeros(amount, dtype='bool_')
    for i in truth_indexes:
        temp_array[i] = True
    # build list containing all training indizes
    train_indexes = np.array(np.array(range(amount))[~temp_array], dtype=int)


    # does parts of the slicing as it is required in all generators
    def slice_fast(data, indexes):
        batch = np.zeros(((len(indexes), t_len) + data.shape[1:]), dtype=data.dtype)
        for i in range(len(indexes)):
            batch[i] = data[indexes[i]: indexes[i] + t_len]
        return batch

    def train_gen():
        print(f'optimum training batches are {int(np.floor(len(train_indexes)/batch_size))}')
        while True:
            # generate radom sequence of the complete set for one epoch
            rans = np.random.choice(len(train_indexes), size=int(np.floor(len(train_indexes)/batch_size)), replace=False)
            # yield one batch until the epoch is done
            for i in range(len(rans)//batch_size - 1):
                yield slice_fast(data, rans[i * batch_size: (i+1) * batch_size]), truth[rans[i * batch_size: (i+1) * batch_size]+t_len].reshape((batch_size, 1) + truth.shape[1:])

    def truth_gen():
        print(f'optimum validation batches are {int(np.ceil(len(truth_indexes)/batch_size))}')
        while True:
            rans = np.zeros(batch_size*int(np.ceil(len(truth_indexes)/batch_size)), dtype=int)
            rans[:len(truth_indexes)] = np.random.choice(len(truth_indexes), size=len(truth_indexes), replace=False)
            rans[len(truth_indexes):] = np.random.choice(len(truth_indexes), size=len(rans)-len(truth_indexes), replace=False)
            for i in range(int(np.ceil(len(truth_indexes)/batch_size))-1):
                yield slice_fast(data, rans[i * batch_size: (i+1) * batch_size]), truth[rans[i * batch_size: (i+1) * batch_size]+t_len].reshape((batch_size, 1) + truth.shape[1:])

    def comp_gen():
        while True:
            for i in range(len(truth_indexes)//batch_size):
                yield slice_fast(data, truth_indexes[i * batch_size: (i+1) * batch_size]), truth[truth_indexes[i * batch_size: (i+1) * batch_size]+t_len].reshape((batch_size, 1) + truth.shape[1:])

    return train_gen, truth_gen, comp_gen, truth_indexes[:(len(truth_indexes)//batch_size)*batch_size]




#build_generators(np.zeros(50), None, 0.5, 666, 32)

