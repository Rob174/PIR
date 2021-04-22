from time import strftime, gmtime, localtime
import os


class FolderInfos:
    data_folder = None
    base_folder = None
    base_filename = None
    id = None

    @staticmethod
    def init(custom_name="",subdir=""):
        FolderInfos.id = strftime("%Y-%m-%d_%Hh%Mmin%Ss", localtime())
        FolderInfos.data_folder = "/".join(os.path.realpath(__file__).split("/")[:-3] + ["data/"])
        if subdir != "":
            FolderInfos.data_folder += subdir +"/" if subdir[-1] != "/" else subdir
        FolderInfos.base_folder = FolderInfos.data_folder + FolderInfos.id + "_"+custom_name+"/"
        FolderInfos.base_filename = FolderInfos.base_folder + FolderInfos.id+"_"+custom_name
        os.mkdir(FolderInfos.base_folder)