from typing import Dict, List

from model_keras.FolderInfos import FolderInfos
import  tensorflow as tf


def create_summary(writer: tf.summary.SummaryWriter,optimizer_name: str,optimizer_parameters: Dict,
                   loss: str,metriques_utilisees: List[str],
                   but_essai: str,informations_additionnelles: str, id: str,dataset_name: str="",taille_x_img: int=1600,
                   taille_y_img: int=900, batch_size=10,
                   nb_img_tot=173959,nb_epochs=1,nb_tr_img=36325):
    markdown = f"""# Résumé de l'entrainement du {id}

Entrainement sur {dataset_name} avec pour l'entrainement {nb_tr_img} ({nb_img_tot} images au total) avec des images de taille {taille_x_img} px par {taille_y_img} px
Batch size de {batch_size}


## Paramètres d'entrainement

Entrainement sur {nb_epochs} epochs

Optimisateur {optimizer_name} avec les paramètres :\n"""
    for k,v in optimizer_parameters.items():
        markdown += f"{k} : {v}"
    markdown += f"""\nLoss : {loss}

Métriques : """
    markdown += ", ".join([f"{metrique}" for metrique in metriques_utilisees])
    markdown += f"""\n## Description de l'essai\n\n{but_essai}\n\n{informations_additionnelles}"""

    with writer.as_default():
        tf.summary.text("Resume", markdown, step=0)
        writer.flush()



