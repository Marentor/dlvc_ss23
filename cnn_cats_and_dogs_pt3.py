import numpy as np
import torch
import torch.nn as nn

from dlvc import ops
from dlvc.batches import BatchGenerator
from dlvc.datasets.pets import PetsDataset
from dlvc.models.pytorch import CnnClassifier
from dlvc.test import Accuracy
from cnn_cats_and_dogs import Cat_and_dog_classifier


def train_and_test(name: str, op: ops, lr: float, wd: float):
    train = PetsDataset(r"C:\Users\hp\Documents\Uni\Master\Semester_4\VCDL\cifar-10-batches-py\\", 1)
    val = PetsDataset(r"C:\Users\hp\Documents\Uni\Master\Semester_4\VCDL\cifar-10-batches-py\\", 2)
    test = PetsDataset(r"C:\Users\hp\Documents\Uni\Master\Semester_4\VCDL\cifar-10-batches-py\\", 3)

    size_of_batch = 128
    gen_train = BatchGenerator(dataset=train, num=size_of_batch, shuffle=True, op=op)
    gen_val = BatchGenerator(dataset=val, num=size_of_batch, shuffle=True, op=op)
    gen_test = BatchGenerator(dataset=test, num=size_of_batch, shuffle=True, op=op)

    best_test_acc = 0
    all_val_acc = []
    val_acc_per_epoch = []
    loss_per_epoch = []

    best_model = None
    clf = CnnClassifier(Cat_and_dog_classifier(), (0, 32, 32, 3), 2, lr, wd)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    clf.net.to(device)
    for epoch in range(100):
        losses = []
        for batch in gen_train:
            loss = (clf.train(batch.data, batch.label))
            losses.append(loss)

        for val_batch in gen_val:
            output = clf.predict(val_batch.data)
            acc = Accuracy()
            acc.update(output.detach().numpy(), val_batch.label)
            all_val_acc.append(acc.accuracy())
            if acc.accuracy() > best_test_acc:
                best_model = clf.net.state_dict()

        losses = np.array(losses)
        print(
            f"Epoch: {epoch} \nTrain loss: {np.mean(losses)} + std:{np.std(losses)} \nValidation accuracy {acc.accuracy()}\n")
        val_acc_per_epoch.append(acc.accuracy())
        loss_per_epoch.append(np.mean(losses))


    #test the best model on the test set
    model = CnnClassifier(Cat_and_dog_classifier(), (0, 32, 32, 3), 2, lr, wd)
    model.net.load_state_dict(best_model)
    for test_batch in gen_test:
        output = model.predict(test_batch.data)
        acc = Accuracy()
        acc.update(output.detach().numpy(), test_batch.label)
    print("BEST MODEL\nValidation accuracy {val}\nTest accuracy: {test}".format(val=max(all_val_acc),
                                                                              test=acc.accuracy()))
    best_test_acc = acc.accuracy()

    # writes out the model to  best_model.pt
    torch.save(model.net.state_dict(), "{}.pt".format(name))

    for train_batch in gen_train:
        output = model.predict(train_batch.data)
        acc = Accuracy()
        acc.update(output.detach().numpy(), train_batch.label)

    train_acc = acc.accuracy()

    #dic_to_write = {"name": name, "best_test_acc": best_test_acc,
    #                "val_acc_per_epoch": val_acc_per_epoch, "loss_per_epoch": loss_per_epoch, "train_acc": train_acc

    # writes out the dictionary to  model_performances.txt
    #with open("model_performances.txt", "a") as f:
    #    f.write(str(dic_to_write) + "\n")



standard_op = ops.chain([
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1 / 127.5),
    ops.hwc2chw()
])

crop_op = ops.chain([
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1 / 127.5),
    ops.rcrop(32, 4, "constant"),
    ops.hwc2chw()
    ])

flip_op = ops.chain([
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1 / 127.5),
    ops.hflip(),
    ops.hwc2chw()
    ])

crop_flip_op = ops.chain([
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1 / 127.5),
    ops.rcrop(32, 4, "constant"),
    ops.hflip(),
    ops.hwc2chw()
    ])

less_crop_flip_op = ops.chain([
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1 / 127.5),
    ops.rcrop(32, 2, "constant"),
    ops.hflip(),
    ops.hwc2chw()
    ])

one_p_crop_flip_op = ops.chain([
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1 / 127.5),
    ops.rcrop(32, 1, "constant"),
    ops.hflip(),
    ops.hwc2chw()
])


if __name__ == "__main__":
    #train_and_test("Baseline", standard_op, 0.01, 0)
    #train_and_test("flip", flip_op, 0.01, 0)
    #train_and_test("crop", crop_op, 0.01, 0)
    #train_and_test("crop_flip", crop_flip_op, 0.01, 0)
    #train_and_test("weight_decay_low", standard_op, 0.01, 0.001)
    #train_and_test("weight_decay_high", standard_op, 0.01, 0.1)
    #train_and_test("crop_flip_weight_decay_low", crop_flip_op, 0.01, 0.001)
    #train_and_test("crop_flip_weight_decay_high", crop_flip_op, 0.01, 0.1)
    #train_and_test("flip_weight_decay_high", flip_op, 0.01, 0.1)
    train_and_test("less_crop_flip_weight_decay_high", less_crop_flip_op, 0.01, 0.1)
    #train_and_test("1p_crop_flip_weight_decay_high", one_p_crop_flip_op, 0.01, 0.1)


