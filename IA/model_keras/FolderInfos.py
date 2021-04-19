from time import strftime, gmtime, localtime
import os


class FolderInfos:
    base_folder = None
    base_filename = None
    id = None

    @staticmethod
    def init(custom_name=""):
        FolderInfos.id = strftime("%Y-%m-%d_%Hh%Mmin%Ss", localtime())
        FolderInfos.base_folder = "/".join(os.path.realpath(__file__).split("/")[:-3] + ["data/"]) + FolderInfos.id + "_"+custom_name+"/"
        FolderInfos.base_filename = FolderInfos.base_folder + FolderInfos.id+"_"+custom_name
        os.mkdir(FolderInfos.base_folder)
