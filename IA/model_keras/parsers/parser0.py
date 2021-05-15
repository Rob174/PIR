import argparse


class Parser0:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.args = {
                    '-img_w':['image_width',532,int,"Indique la gpu visible par le script tensorflow"],
                    '-bs':['batch_size',10,int,"Indique le nombre d'images par batch"],
                    '-gpu':['gpu_selected',"0",str,"Indique la gpu visible par le script tensorflow"],
                    '-lr':['lr',1e-3,float,"Indique la gpu visible par le script tensorflow"],
                    '-eps':['epsilon',1e-7,float,"Indique la gpu visible par le script tensorflow"],
                    '-opti':['optimizer',"adam",str,"Optimisateur"],
                    '-nbImg':['nb_images',1080,int,"Indique le nb d'imgs passées en entrainement avant l'arrêt"],
                    '-tailleMini':['taille_mini_obj_px',10,int,"Indique la taille minimale des objets que doit détecter le réseau (après redimensionnement)"],
                    '-nbEpochs':['nb_epochs',1,int,"Indique le nb de passage du dataset"],
                    '-augm':['augmentation',"f",str,"Indique le nb de passage du dataset"]}

    def __call__(self, *args, **kwargs):
        for arg_name,[variable,default_val,type,description] in self.args.items():
            self.parser.add_argument(arg_name,dest=variable,default=default_val,type=type,help=description)
        return self.parser.parse_args()