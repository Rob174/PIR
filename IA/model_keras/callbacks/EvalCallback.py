from typing import List

import tensorflow as tf


class EvalCallback(tf.keras.callbacks.Callback):

    def __init__(self, tb_callback: tf.keras.callbacks.Callback, dataset: tf.data.Dataset, batch_size: int,
                 metrics_name: List[str], eval_rate: int = None, type: str = "tr"):
        self.tb_callback = tb_callback
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
            (batch_imgs, batch_labels) = self.dataset.take(1)
            test_output = self.model.test(batch_imgs, batch_labels)

            items_to_write = {}

            for metric, metric_value in zip(self.metrics_names, test_output):
                items_to_write[metric] = metric_value

            writer = self.tb_callback.writer
            for name, value in items_to_write.items():
                summary = tf.summary.scalar(self.type + name, value, step=self.step_number)
                writer.add_summary(summary, self.step_number)
                writer.flush()
        self.step_number += self.batch_size
