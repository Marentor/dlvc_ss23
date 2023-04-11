import pickle

import numpy as np

from ..dataset import Sample, Subset, ClassificationDataset


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


class PetsDataset(ClassificationDataset):
    """
    Dataset of cat and dog images from CIFAR-10 (class 0: cat, class 1: dog).
    """

    def __init__(self, fdir: str, subset: Subset):
        """
        Loads a subset of the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all cat and dog images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all cat and dog images from "data_batch_5".
          - The test set contains all cat and dog images from "test_batch".

        Images are loaded in the order the appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in BGR channel order.
        """

        cifar_data = []
        cifar_labels = []
        if subset == 1:
            for i in range(1, 5):
                cifar_data_dict = unpickle(fdir + "/data_batch_{}".format(i))
                if i == 1:
                    cifar_data = cifar_data_dict[b"data"]
                    cifar_labels = cifar_data_dict[b"labels"]
                else:
                    cifar_data = np.vstack((cifar_data, cifar_data_dict[b"data"]))
                    cifar_labels.extend(cifar_data_dict[b"labels"])

        if subset == 2:
            cifar_data_dict = unpickle(fdir + "/data_batch_5")
            cifar_data = cifar_data_dict[b"data"]
            cifar_labels = cifar_data_dict[b"labels"]

        if subset == 3:
            cifar_data_dict = unpickle(fdir + "/test_batch")
            cifar_data = cifar_data_dict[b"data"]
            cifar_labels = cifar_data_dict[b"labels"]

        cifar_data = cifar_data.reshape((len(cifar_data), 3, 32, 32))
        cifar_data = np.rollaxis(cifar_data, 1, 4)
        cifar_data = cifar_data[:, :, :, ::-1]

        cifar_labels = np.array(cifar_labels)
        # Locate position of labels that equal to i
        pos_i = np.argwhere((cifar_labels == 3) | (cifar_labels == 5))
        # Convert the result into a 1-D list
        pos_i = list(pos_i[:, 0])
        # Collect all data that match the desired label
        cifar_data = np.array([cifar_data[j] for j in pos_i])
        cifar_labels = np.array([cifar_labels[j] for j in pos_i])

        cifar_labels = np.where(cifar_labels < 4, 0, 1)
        self.cifar_train_data = cifar_data
        self.cifar_train_labels = cifar_labels

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """

        return len(self.cifar_train_labels)

    def __getitem__(self, idx: int) -> Sample:
        """
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds. Negative indices are not supported.
        """

        if idx < 0 or idx > len(self.cifar_train_labels):
            raise IndexError

        return Sample(idx, self.cifar_train_data[idx], self.cifar_train_labels[idx])

    def num_classes(self) -> int:
        """
        Returns the number of classes.
        """

        return len(np.unique(self.cifar_train_labels))
