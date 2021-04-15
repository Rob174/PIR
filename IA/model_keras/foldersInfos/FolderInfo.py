from time import strftime, gmtime
import os
class FolderInfos:
    base_folder = None
    base_filename = None
    params_file = None
    @staticmethod
    def init():
        id = strftime("%Y-%m-%d_%Hh%Mmin%Ss", gmtime())
        FolderInfos.base_folder= "/".join(os.path.realpath(__file__).split("/")[:-2]+["data"])+id+"/"
        FolderInfos.base_filename = FolderInfos.base_folder + id
        FolderInfos.params_file = FolderInfos.base_filename + "parametres_entrainement.json"
        os.mkdir(FolderInfos.base_folder)
