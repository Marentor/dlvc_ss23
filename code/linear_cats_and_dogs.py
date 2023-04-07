import dlvc.ops as ops
import numpy as np
import torch
from code.dlvc.test import Accuracy
from code.dlvc.batches import BatchGenerator
from code.dlvc.datasets.pets import PetsDataset


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)
        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, x):
       x = self.linear(x)
       return self.soft(x)


op = ops.chain([
    ops.vectorize(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1 / 127.5),
])

train = PetsDataset(r"C:\Users\hp\Documents\Uni\Master\Semester_4\VCDL\cifar-10-batches-py", 1)
val = PetsDataset(r"C:\Users\hp\Documents\Uni\Master\Semester_4\VCDL\cifar-10-batches-py", 2)
test = PetsDataset(r"C:\Users\hp\Documents\Uni\Master\Semester_4\VCDL\cifar-10-batches-py", 3)

gen_train = BatchGenerator(dataset=train, num=len(train), shuffle=True, op=op)
gen_val = BatchGenerator(dataset=val, num=len(val), shuffle=True, op=op)
gen_test = BatchGenerator(dataset=test, num=len(test), shuffle=True, op=op)

model = LinearClassifier(input_dim=np.prod(train.cifar_train_data.shape[1:4]), num_classes=train.num_classes())
loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters())
'''
TODO: Train a model for multiple epochs, measure the classification accuracy on the validation dataset throughout the training 
and save the best performing model. 
After training, measure the classification accuracy of the best perfroming model on the test dataset. 
Document your findings in the report.
'''
best_val_acc = 0
all_val_acc = []
best_model = None
for epoch in range(100):
    for batch in gen_train:

        optimizer.zero_grad()

        output = model.forward(torch.from_numpy(batch.data))

        loss = loss_f(output, torch.from_numpy(batch.label) )
        loss.backward()

        optimizer.step()

    for val_batch in gen_val:
        output = model.forward(torch.from_numpy(val_batch.data))
        acc = Accuracy()
        acc.update(output.detach().numpy(),val_batch.label)
        all_val_acc.append(acc.accuracy())
        if acc.accuracy() > best_val_acc:
            best_model = model.state_dict()

    #only uses loss of last batch (but should matter for this exercise)
    print("Epoch: {epoch} \nTrain loss: {trainloss} \nValidation accuracy {accuracy}\n".format(epoch=epoch, trainloss=loss, accuracy=acc.accuracy()))

model =  LinearClassifier(input_dim=np.prod(train.cifar_train_data.shape[1:4]), num_classes=train.num_classes())
model.load_state_dict(best_model)
for test_batch in gen_test:
    output = model.forward(torch.from_numpy(test_batch.data))
    acc = Accuracy()
    acc.update(output.detach().numpy(), test_batch.label)
print("BEST MODEL\nValidation accuracy {val}\nTest accuracy: {test}".format(val=max(all_val_acc),test=acc.accuracy()))






