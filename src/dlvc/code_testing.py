from src.dlvc.datasets.pets import PetsDataset
#from src.dlvc.ops import chain,vectorize,type_cast,add,mul
from src.dlvc import ops
from src.dlvc.batches import BatchGenerator
import numpy as np
p=PetsDataset(r"C:\Users\hp\Documents\Uni\Master\Semester_4\VCDL\cifar-10-batches-py",1)
p2=PetsDataset(r"C:\Users\hp\Documents\Uni\Master\Semester_4\VCDL\cifar-10-batches-py",2)
p3=PetsDataset(r"C:\Users\hp\Documents\Uni\Master\Semester_4\VCDL\cifar-10-batches-py",3)

#Op = Callable[[np.ndarray], np.ndarray]

op_ = ops.chain([
    ops.vectorize(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1/127.5),
])

gen = BatchGenerator(dataset=p, num= 500, shuffle=False, op=op_)

for idx, i in enumerate(gen):
    print("Batch number:",idx)
    #print(i.label)
    #print(i.data)
    #print(i.idx)

