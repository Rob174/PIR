import argparse

from IA.model_keras.parsers.parser0 import Parser0


class Parser1(Parser0):
    def __init__(self):
        super(Parser1, self).__init__()
        self.args = dict(self.args,
                         **{
                    '-img_w':['image_width',532,int,"Indique la gpu visible par le script tensorflow"],
                    '-lastAct':['lastActivation',"linear",str,"Indique la gpu visible par le script tensorflow"],
                    '-approxAccur':['approximationAccuracy',"none",str,"Indique la gpu visible par le script tensorflow"],
                    '-classWeigths':['classes_weights',"False",str,"Type de pondération des classes sous-représentées"],
                    })
