import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from model_keras.foldersInfos.FolderInfo import FolderInfos

matplotlib.use('Agg')

class SimpleAnalyser:
    work_directory = None
    def __init__(self, eval_frequency,attribut_entrainement, model):
        """
        # In:

        - **work_directory** : *str* le chemin du dossier dans lequel mettre les résultats
        - **trainer_param** : *dict* les paramètres d'entrainement sous forme d'un dictionnaire
        - **model** : *kerass Model* le model que l'on entraine
        - **nom_model** : *str* le nom du modèle à indiquer sur le nom du dossier
        """
        self.model = model


        # Préparation de la liste pour sauvegarder les valeurs d'entrainement
        self.valeurs_entrainement = []
        self.eval_frequency = eval_frequency
        self.attribut_entrainement = attribut_entrainement

    def save_step(self, dictionnaire_sauvegarde):
        """Sauvegarde les résultats d'un test"""
        self.valeurs_entrainement.append(dictionnaire_sauvegarde)
        processus_async = multiprocessing.Process(target=plot_errors,args=(self.valeurs_entrainement,
                                                                           self.eval_frequency,
                                                                           self.attribut_entrainement,
                                                                           FolderInfos.work_directory,
                                                                           FolderInfos.identify_machine()))
        processus_async.daemon = True  # On n'attend pas la fin de l'exécution de la fonction
        processus_async.start()

    def save_point_arret(self, nb_it_tot, nb_it_par_epoch):
        with SaveWriter(FoldersInfos.params_file) as data:
            data["entrainement"] = {"nb_iterations_realisees":nb_it_tot,"nb_iterations_par_epoch" : nb_it_par_epoch}

    def save_model(self, prct):
        """Sauvegarde du modèle à une étape d'entrainement"""
        # os.system("rm %s/*model*.h5" % FoldersInfos.work_directory)
        self.model.model.keras_layer.save(FoldersInfos.work_directory  + "model_%03d_prct.h5" % (prct))

    def write_results(self, valid_error, valid_variance, valid_error_prct, mat_conf):
        """Enregistre les valeurs de précision finale et écrit le tout dans un fichier"""
        # On finit de constituer le fichier markdown
        with SaveWriter(FoldersInfos.params_file) as data:
            data["entrainement"]["valid_error_prct"] = valid_error
            data["entrainement"]["valid_variance"] = valid_variance
            data["entrainement"]["valid_error"] = valid_error_prct
        plot_errors(valeurs_entrainement=self.valeurs_entrainement,eval_frequency=self.eval_frequency,
                    attribut_entrainement=self.attribut_entrainement,work_directory=FoldersInfos.work_directory,nom_machine=FoldersInfos.identify_machine())
        self.plot_confusion_matrix(mat_conf, titre="Confusion CNN\&Inception-pngx16",
                                   ticks=[ str(i) for i in np.arange(0, len(mat_conf[0]))] + ["total","%"])
        # On enregistre les courbes d'erreurs
        with open("%sresultats_entrainement.json" % FoldersInfos.work_directory, 'w', encoding="utf8") as f:
            liste_dico_adapte = []
            for dico in self.valeurs_entrainement:
                liste_dico_adapte.append({k: str(v) for k, v in dico.items()})
            json.dump(liste_dico_adapte,
                      f)  # Utilisé uniquement (pour le moment) dans le notebook d'analyse des résultats

    def save_final_time(self, time):
        """
        # In:
        - **time** : temps d'entrainement en secondes
        """
        with SaveWriter(FoldersInfos.params_file) as data:
            heures = int(time / 3600)
            min = int((time - heures * 3600) / 60)
            sec = int((time - heures * 3600 - min * 60))
            data["entrainement"]["tr_time"] = "%d h %d min %d sec\n" % (heures, min, sec)


    def analyse_save_eval_labels_pred(self, labels, pred):
        """ Analyse les labels et prédiction et les sauvegarde
        # In:
            - **labels** : *np.ndarray* de forme [batches_dim,pred_dim] : PDF des labels demandés
            - **pred** : *np.ndarray* de forme [batches_dim,pred_dim] : PDF des probas de chaque label
        """
        # Pour sauvegarder les labels et prédictions on utilisera np.save et un fichier npy par array
        np.save(FoldersInfos.work_directory + "labels.npy", labels)
        np.save(FoldersInfos.work_directory + "pred.npy", pred)

    def plot_confusion_matrix(self, orig_conf_matrix, titre, ticks, r=0, fs=10, dilat=1, cm='PuBuGn'):
        """
        Entrées:
            - mat_cf {array/matrix} : votre matrice de confusion
            - titre {string} : le titre
            - ticks {iterable} : les noms des classes qui apparaîtront sur le côté de la matrice
            - r {int,opt} : facteur de réduction
            - fs {int, opt} : taille de police
            - dilat {float, opt} : coefficient multiplicateur (ajuste la couleur)
            - cm {string, opt} : matplotlib.pyplot colormap
        Sorties:
            - {None} : affiche la matrice de confusion dans une figure plt
            - kia {float} : coefficient "kappa" de "chance agreement"
            - pcb {int} : pourcentage de bien classés
        """
        nb_classes = len(orig_conf_matrix[0])
        maxtrix_with_totals = np.zeros((nb_classes + 2, nb_classes + 2))
        # Normalisation des valeurs par ligne
        for j in range(nb_classes):
            if np.sum(orig_conf_matrix[j, :]) > 0:
                maxtrix_with_totals[j, :-2] = orig_conf_matrix[j, :] / np.sum(orig_conf_matrix[j, :])

        pc = np.dot(np.sum(orig_conf_matrix, 0), np.sum(orig_conf_matrix, 1))
        total_echantillons = np.sum(orig_conf_matrix)
        kia = (total_echantillons * np.sum(np.diag(orig_conf_matrix)) - pc) / (total_echantillons * total_echantillons - pc)
        prct_bien_classes = (np.sum(np.diag(orig_conf_matrix)) / total_echantillons) * 100
        # Pourcentage classifications correctes en comptant les 3 diagonales "centrales"
        pbc_ktop = np.sum([s for s in map(np.sum,
                [np.diag(orig_conf_matrix, -1), np.diag(orig_conf_matrix), np.diag(orig_conf_matrix, 1)]
                                           )]) / total_echantillons * 100
        # On constitue la matrice de confusion finale avec les totaux à afficher (-2 : compte ; -1 : en prcts)
        final_conf_matrix = np.zeros((nb_classes + 2, nb_classes + 2))
        final_conf_matrix[:-2, :-2] = orig_conf_matrix
        final_conf_matrix[:-2, -2] = np.sum(orig_conf_matrix, axis=1)
        # prct .... correct
        final_conf_matrix[:-2, -1] = 100 * np.diag(orig_conf_matrix) / final_conf_matrix[:-2, -2] # Si nul notera nd pour le pourcentage
        # VdL quand np.sum(m, axis = 1), genere une erreur et ""nd" comme taux de classement
        final_conf_matrix[-2, :-2] = np.sum(orig_conf_matrix, axis=0)
        # prst .... correct
        print(final_conf_matrix[-2, :-2])
        final_conf_matrix[-1, :-2] = 100 * np.diag(orig_conf_matrix) / final_conf_matrix[-2, :-2] # Si nul notera nd pour le pourcentage
        # VdL quand np.sum(m, axis = 0), genere une erreur et ""nd" comme taux de classement
        final_conf_matrix[-2, -2] = np.sum(np.diag(orig_conf_matrix))
        final_conf_matrix[-1, -1] = 100 * np.sum(np.diag(orig_conf_matrix)) / total_echantillons
        # VdL necessaire pour eviter d'avoir la matrice de confusion en petit comme les erreurs
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        texte = titre + " kappa = %.3f top3voisins = %.1f %%" % (kia, pbc_ktop)
        ax.set_title("\n".join(wrap(texte, 60)))
        ax.set_xlabel('Classe prédite', labelpad=15)
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('Classe étiquetée (vraie)', labelpad=30)
        plt.set_cmap(cm)
        plt.xticks(range(len(ticks)), ticks, fontsize=fs)
        plt.yticks(range(len(ticks)), ticks, fontsize=fs)
        plt.imshow(maxtrix_with_totals * dilat, interpolation="none")
        plt.colorbar()

        fig.tight_layout()
        for x, y in np.ndindex(len(maxtrix_with_totals), len(maxtrix_with_totals)):
            if isnan(final_conf_matrix[y, x]):
                plt.text(x, y, str("nd"), horizontalalignment='center', fontsize=fs)
            else:
                plt.text(x, y, "%.f" % final_conf_matrix[y, x], horizontalalignment='center', fontsize=fs)
        plt.axis('image')
        plt.savefig(FoldersInfos.work_directory + "matrice_confusion.png")
        return prct_bien_classes, kia, pbc_ktop



def plot_errors(valeurs_entrainement):
    fig, axe_error = plt.subplots()
    loss_axe = axe_error.twinx()
    loss_axe.plot(
        np.array(Lcoordx_tr) * dataset.batch_size,
        liste_lossTr, color="r", label="lossTr")
    loss_axe.plot(np.array(Lcoordx_valid) * dataset.batch_size, liste_lossValid, color="orange", label="lossValid")
    axe_error.plot(np.array(Lcoordx_tr) * dataset.batch_size, 100 * (1 - np.array(liste_accuracyTr)), color="g",
                   label="tr_error")
    axe_error.plot(np.array(Lcoordx_valid) * dataset.batch_size, 100 * (1 - np.array(liste_accuracyValid)), color="b",
                   label="valid_error")
    axe_error.set_xlabel("Nombre d'itérations, d'images passées")
    axe_error.set_ylabel("Error (%)")
    loss_axe.set_ylabel("Loss (MSE)")
    fig.legend()
    plt.grid()
    plt.savefig("/home/rmoine/Documents/erreur_accuracy_batch_size_%d.png" % dataset.batch_size)
    plt.clf()
    plt.close(fig)