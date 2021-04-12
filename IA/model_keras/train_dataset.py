
from data.generate_data import Nexet_dataset

dataset = Nexet_dataset()

# model=make_model((1600,900),num_classes=len(dataset.correspondances_classes.keys()))
for epochs in range(1):
    iterator=dataset.getNextBatch()
    while True:
        try:
            batchImg, batchLabel =next(iterator)
            print(batchImg.shape, batchLabel.shape)
        except StopIteration:
            print("Epoch %d done" % epochs)
            break
