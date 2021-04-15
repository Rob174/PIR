import argparse


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-bs', dest='batch_size', default=10, type=int,
                        help="[Optionnel] Indique le nombre d'images par batch")
    parser.add_argument('-gpu', dest='gpu_selected', default="0", type=str,
                        help="[Optionnel] Indique la gpu visible par le script tensorflow")
    parser.add_argument('-img_w', dest='image_width', default=400, type=int,
                        help="[Optionnel] Indique la gpu visible par le script tensorflow")
    args = parser.parse_args()
    return args
