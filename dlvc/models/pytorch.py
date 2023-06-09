import numpy as np
import torch
import torch.nn as nn

from ..model import Model

class CnnClassifier(Model):
    '''
    Wrapper around a PyTorch CNN for classification.
    The network must expect inputs of shape NCHW with N being a variable batch size,
    C being the number of (image) channels, H being the (image) height, and W being the (image) width.
    The network must end with a linear layer with num_classes units (no softmax).
    The cross-entropy loss (torch.nn.CrossEntropyLoss) and SGD (torch.optim.SGD) are used for training.
    '''

    def __init__(self, net: nn.Module, input_shape: tuple, num_classes: int, lr: float, wd: float):
        '''
        Ctor.
        net is the cnn to wrap. see above comments for requirements.
        input_shape is the expected input shape, i.e. (0,C,H,W).
        num_classes is the number of classes (> 0).
        lr: learning rate to use for training (SGD with e.g. Nesterov momentum of 0.9).
        wd: weight decay to use for training.
        '''



        # Inside the train() and predict() functions you will need to know whether the network itself
        # runs on the CPU or on a GPU, and in the latter case transfer input/output tensors via cuda() and cpu().
        # To termine this, check the type of (one of the) parameters, which can be obtained via parameters() (there is an is_cuda flag).
        # You will want to initialize the optimizer and loss function here.
        # Note that PyTorch's cross-entropy loss includes normalization so no softmax is required

        self.net = net
        self.input = input_shape
        self.num_classes = num_classes
        self.lr = lr
        self.wd = wd

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)

    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple.
        '''

        return self.input

    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''

        return (self.num_classes,)

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on batch of data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns the training loss.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''


        # Make sure to set the network to train() mode
        # See above comments on CPU/GPU

        if not isinstance(data, np.ndarray) or data.dtype != np.float32 or data.ndim != 4:
            raise TypeError("data must be a NumPy array of shape (m,C,H,W) and type np.float32")
        if not isinstance(labels, np.ndarray) or labels.dtype != np.int64 or labels.ndim != 1:
            raise TypeError("labels must be a NumPy array of shape (m,) and type np.int64")

        if data.shape[0] != labels.shape[0]:
            raise ValueError("data and labels must have the same number of samples")
        if not (0 <= labels.min() and labels.max() < self.num_classes):
            raise ValueError("labels must have integral values between 0 and num_classes - 1")

        data = torch.from_numpy(data)
        labels = torch.from_numpy(labels)
        if next(self.net.parameters()).is_cuda:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        # Transfer to GPU
        data, labels = data.to(device), labels.to(device)

        self.net.train()
        self.optimizer.zero_grad()
        outputs = self.net(data.float())
        # Compute the loss and its gradients
        loss = self.criterion(outputs, labels.long())
        # Backward pass
        loss.backward()

        # Update the parameters
        self.optimizer.step()

        return loss.item()


    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict softmax class scores from input data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        The scores are an array with shape (n, output_shape()).
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''



        # Pass the network's predictions through a nn.Softmax layer to obtain softmax class scores
        # Make sure to set the network to eval() mode
        # See above comments on CPU/GPU

        if not isinstance(data, np.ndarray) or data.dtype != np.float32 or data.ndim != 4:
            raise TypeError("x argument must be a 4D numpy array with dtype np.float32")

        self.net.eval()
        data = torch.from_numpy(data)
        if next(self.net.parameters()).is_cuda:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        # Transfer to GPU
        data = data.to(device)
        predictions = self.net(data)
        scores = torch.softmax(predictions, dim=1)
        return  scores.cpu()
