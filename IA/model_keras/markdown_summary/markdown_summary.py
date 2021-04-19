from typing import Dict, List

from model_keras.FolderInfos import FolderInfos
import  tensorflow as tf


def create_summary(writer: tf.summary.SummaryWriter,optimizer_name: str,optimizer_parameters: Dict,
                   loss: str,metriques_utilisees: List[str],
                   but_essai: str,informations_additionnelles: str, model_img_path: str, id: str):
    markdown = f"""# Résumé de l'entrainement du {id}

## Paramètres d'entrainement

Optimisateur {optimizer_name} avec les paramètres :\n"""
    for k,v in optimizer_parameters.items():
        markdown += f"{k} : {v}"
    markdown += f"""\nLoss : {loss}

Métriques : """
    markdown += ", ".join([f"{metrique}" for metrique in metriques_utilisees])
    markdown += f"""\n## Description de l'essai\n\n{but_essai}\n\n{informations_additionnelles}"""

    markdown += f"\n\n## Architecture du modèle\n\n![Modele]({model_img_path}"
    with writer.as_default():
        tf.summary.text("Resume", markdown, step=0)
        writer.flush()



