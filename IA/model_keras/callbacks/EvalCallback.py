from typing import List

import tensorflow as tf


class EvalCallback(tf.keras.callbacks.Callback):

    def __init__(self, writer: tf.summary.SummaryWriter, dataset: tf.data.Dataset, batch_size: int,
                 metrics_name: List[str], eval_rate: int = None, type: str = "tr"):
        super().__init__()
        self.writer = writer
        self.step_number = 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.metrics_names = metrics_name
        if type == "tr":
            self.eval_rate = 1
        else:
            self.eval_rate = eval_rate
        self.type = type

    def on_train_batch_end(self, batch, logs=None):
        if self.step_number % self.eval_rate == 0:
            batch = list(self.dataset.take(1).as_numpy_iterator())[0]
            print(len(batch))
            [batch_imgs, batch_labels] = batch
            test_output = self.model.test_on_batch(batch_imgs, batch_labels)

            items_to_write = {}
            for i,[metric, metric_value] in enumerate(zip(self.metrics_names, test_output)):
                if i > 0:
                    items_to_write[metric] = 100*(1.-metric_value)
                else:
                    items_to_write[metric] = metric_value

            with self.writer.as_default():
                for name, value in items_to_write.items():
                    tf.summary.scalar(self.type + name, value, step=self.step_number)
                    self.writer.flush()
        self.step_number += self.batch_size
