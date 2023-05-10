# Deep Learning for Visual Computing - Assignment 2

The second assignment covers iterative optimization and parametric (deep) models for image classification.

## Part 1

This part is about experimenting with different flavors of gradient descent and optimizers.

Download the data from [here](https://smithers.cvl.tuwien.ac.at/jstrohmayer/dlvc_ss23/-/tree/main/assignments/assignment_2). Your task is to implement `optimizer_2d.py`. We will use various optimization methods implemented in PyTorch to find the minimum in a 2D function given as an image. In this scenario, the optimized weights are the coordinates at which the function is evaluated, and the loss is the function value at those coordinates.

See the code comments for instructions. The `fn/` folder contains sampled 2D functions for use with that script. For bonus points you can add and test your own functions (something interesting with a few local minima). For this you don't necessarily have to use `load_image`, you can also write a different function that generates a 2D array of values.

The goal of this part is for you to better understand the optimizers provided by PyTorch by playing around with them. Try different types (SGD, Adam etc.), parameters, starting points, and functions. How many steps do different optimizers take to terminate? Is the global minimum reached? What happens when weight decay is set to a non-zero value and why? This nicely highlights the function and limitations of gradient descent, which we've already covered in the lecture.

## Part 2

Time for some Deep Learning. We already implemented most of the required functionality during Assignment 1. Make sure to fix any mistakes mentioned in the feedback you received for your submission. With the exception of `linear_cats_and_dogs.py` all files will be reused in this assignment. The main thing that is missing is a subtype of `Model` that wraps a PyTorch CNN classifier. Implement this type, which is defined inside `dlvc/models/pytorch.py` and named `CnnClassifier`. Details are stated in the code comments. The PyTorch documentation of `nn.Module`, which is the base class of PyTorch models, is available [here](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).

PyTorch (and other libraries) expects the channel dimension of a single sample to be the first one, rows the second one, and columns the third one (`CHW` for short). However in our case they are `HWC`. To address this, implement the `hwc2chw()` function in `ops.py` (make sure to download the updated reference code).

Once this is in place, create a script named `cnn_cats_and_dogs.py`. This file will be very similar to the version for the linear classifier (`linear_cats_and_dogs.py`) developed for Assignment 1 so you might want to use that one as a reference. This file should implement the following in the given order:

1. Load the training, validation and test subsets of the `PetsDataset`.
2. Initialize `BatchGenerator`s for both with batch sizes of 128 or so (feel free to experiment) and the input transformations required for the CNN. This should include input normalization. A basic option is `ops.add(-127.5), ops.mul(1/127.5)` but for bonus points you can also experiment with more sophisticated alternatives such as per-channel normalization using statistics from the training set (if so create corresponding operations in `ops.py` and document your findings in the report).
3. Define a PyTorch CNN with an architecture suitable for cat/dog classification. To do so create a subtype of `nn.Module` and overwrite the `__init__()` and `forward()` methods (do this inside `cnn_cats_and_dogs.py`). If you have access to an Nvidia GPU transfer the model using the `.cuda()` method of the CNN object.
4. Wrap the CNN object `net` in a `CnnClassifier`, `clf = CnnClassifier(net, ...)`.
5. Inside a `for epoch in range(100):` loop (i.e. train for 100 epochs which is sufficient for now), train `clf` on the training set and store the losses returned by `clf.train()` in a list. Then convert this list to a numpy array and print the mean and standard deviation in the format `mean ± std`. Then print the accuracy on the validation set using the `Accuracy` class developed in Assignment 1. While training, keep track of the best performing model with respect to validation accuracy and save it. At the end of the run compute the accuracy on the test subset and print it out as well. 

The console output should thus be similar to the following (ignoring the values):
```python
epoch 1
train loss: 0.689 ± 0.006
val acc: 0.561
epoch 2
train loss: 0.681 ± 0.008
val acc: 0.578
epoch 3
train loss: 0.673 ± 0.009
val acc: 0.585
epoch 4
train loss: 0.665 ± 0.013
val acc: 0.594
epoch 5
train loss: 0.658 ± 0.014
val acc: 0.606
--------------------
val acc (best): 0.606
test acc: 0.612
...
```

The goal of this part is for you to get familiar with PyTorch and to be able to try out different architectures and layer combinations. The pets dataset is ideal for this purpose because it is small. Experiment with the model by editing the code manually rather than automatically via hyperparameter optimization. What you will find is that the training loss will approach 0 even with simple architectures (demonstrating how powerful CNNs are and how well SGD works with them) while the validation accuracy will likely not exceed 75%. The latter is due to the small dataset size, resulting in overfitting. We will address this in the next part.
