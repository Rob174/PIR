import argparse


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-bs', dest='batch_size', default=10, type=int,
                        help="[Optionnel] Indique le nombre d'images par batch")
    parser.add_argument('-gpu', dest='gpu_selected', default="0", type=str,
                        help="[Optionnel] Indique la gpu visible par le script tensorflow")
    parser.add_argument('-img_w', dest='image_width', default=400, type=int,
                        help="[Optionnel] Indique la gpu visible par le script tensorflow")
    parser.add_argument('-lr', dest='lr', default=1e-3, type=float,
                        help="[Optionnel] Indique la gpu visible par le script tensorflow")
    parser.add_argument('-eps', dest='epsilon', default=1e-7, type=float,
                        help="[Optionnel] Indique la gpu visible par le script tensorflow")
    parser.add_argument('-lastAct', dest='lastActivation', default="linear", type=str,
                        help="[Optionnel] Indique la gpu visible par le script tensorflow")
    parser.add_argument('-approxAccur', dest='approximationAccuracy', default="none", type=str,
                        help="[Optionnel] Indique la gpu visible par le script tensorflow")
    parser.add_argument('-nbMod', dest='nb_modules', default=4, type=int,
                        help="[Optionnel] Indique la gpu visible par le script tensorflow")
    parser.add_argument('-redLayer', dest='reduction_layer', default="globalavgpool", type=str,
                        help="[Optionnel] Indique la gpu visible par le script tensorflow")
    parser.add_argument('-dptRate', dest='dropout_rate', default="0.5", type=str,
                        help="[Optionnel] Indique la gpu visible par le script tensorflow")
    parser.add_argument('-opti', dest='optimizer', default="adam", type=str,
                        help="[Optionnel] Optimisateur")
    args = parser.parse_args()
    return args