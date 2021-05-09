from IA.model_keras.parsers.parser1 import Parser1


class Parser2(Parser1):
    def __init__(self):
        super(Parser2, self).__init__()
        print(self.args)
        self.args = dict(self.args,
                         **{'-nbMod': ['nb_modules', 4, int, "Indique la gpu visible par le script tensorflow"],
                            '-redLayer': ['reduction_layer', "globalavgpool", str,
                                          "Indique la gpu visible par le script tensorflow"],
                            '-spatAtt': ['spatial_attention', "n", str,
                                         "Indique la gpu visible par le script tensorflow"],
                            '-dptRate': ['dropout_rate', "0.5", str, "Indique la gpu visible par le script tensorflow"],
                            '-regMod': ['regularize_modules', "n", str, "Indique le nb de passage du dataset"],
                            '-activ': ['activation', "relu", str, "Indique le nb de passage du dataset"],
                            "-loss":['loss',"mse",str,"Choix de la loss"]})

