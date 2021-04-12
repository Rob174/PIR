
from data.generate_data import Nexet_dataset

dataset = Nexet_dataset()

# model=make_model((1600,900),num_classes=len(dataset.correspondances_classes.keys()))
iterator=dataset.getNextBatch()
for epochs in range(1):
    for iter in range(dataset.nb_images//dataset.batch_size):
        batchImg, batchLabel =next(iterator)
        print(batchImg.shape, batchLabel.shape)
