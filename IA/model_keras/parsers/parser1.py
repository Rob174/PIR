from IA.model_keras.parsers.base_parser import BaseParser


class Parser1(BaseParser):
    def __init__(self):
        super(Parser1, self).__init__()
        self.args = dict(self.args,
                         **{'-nbMod': ['nb_modules', 4, int, "Indique la gpu visible par le script tensorflow"],
                            '-redLayer': ['reduction_layer', "globalavgpool", str,
                                          "Indique la gpu visible par le script tensorflow"],
                            '-spatAtt': ['spatial_attention', "n", str,
                                         "Indique la gpu visible par le script tensorflow"],
                            '-dptRate': ['dropout_rate', "0.5", str, "Indique la gpu visible par le script tensorflow"],
                            '-regMod': ['regularize_modules', "n", str, "Indique le nb de passage du dataset"],
                            '-activ': ['activation', "relu", str, "Indique le nb de passage du dataset"]})

