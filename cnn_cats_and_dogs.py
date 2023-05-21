import numpy as np
import torch
import torch.nn as nn

from dlvc import ops
from dlvc.batches import BatchGenerator
from dlvc.datasets.pets import PetsDataset
from dlvc.models.pytorch import CnnClassifier
from dlvc.test import Accuracy

op = ops.chain([
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1 / 127.5),
    ops.hwc2chw()
])

class Cat_and_dog_classifier(nn.Module):
    def __init__(self):
        super(Cat_and_dog_classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear = nn.Linear(512 , 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.pool4(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu5(out)

        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu6(out)
        out = self.pool6(out)

        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

if __name__ == "__main__":
    train = PetsDataset(r"C:\Users\hp\Documents\Uni\Master\Semester_4\VCDL\cifar-10-batches-py\\", 1)
    val = PetsDataset(r"C:\Users\hp\Documents\Uni\Master\Semester_4\VCDL\cifar-10-batches-py\\", 2)
    test = PetsDataset(r"C:\Users\hp\Documents\Uni\Master\Semester_4\VCDL\cifar-10-batches-py\\", 3)
    size_of_batch=128
    gen_train = BatchGenerator(dataset=train, num=size_of_batch, shuffle=True, op=op)
    gen_val = BatchGenerator(dataset=val, num=size_of_batch, shuffle=True, op=op)
    gen_test = BatchGenerator(dataset=test, num=size_of_batch, shuffle=True, op=op)

    best_val_acc = 0
    all_val_acc = []

    best_model = None
    clf = CnnClassifier(Cat_and_dog_classifier(), (0, 32, 32, 3), 2, 0.01, 0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    clf.net.to(device)
    for epoch in range(100):
        losses=[]
        for batch in gen_train:
            loss = (clf.train(batch.data, batch.label))
            losses.append(loss)

        for val_batch in gen_val:
            output = clf.predict(val_batch.data)
            acc = Accuracy()
            acc.update(output.detach().numpy(),val_batch.label)
            all_val_acc.append(acc.accuracy())
            if acc.accuracy() > best_val_acc:
                best_model = clf.net.state_dict()

        losses=np.array(losses)
        print(f"Epoch: {epoch} \nTrain loss: {np.mean(losses)} + mean:{np.std(losses)} \nValidation accuracy {acc.accuracy()}\n")

    model = CnnClassifier(Cat_and_dog_classifier(), (0, 32, 32, 3), 2, 0.01, 0)
    model.net.load_state_dict(best_model)
    for test_batch in gen_test:
        output = model.predict(test_batch.data)
        acc = Accuracy()
        acc.update(output.detach().numpy(), test_batch.label)
    print("BEST MODEL\nValidation accuracy {val}\nTest accuracy: {test}".format(val=max(all_val_acc),test=acc.accuracy()))