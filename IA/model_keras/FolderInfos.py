from time import strftime, gmtime
import os


class FolderInfos:
    base_folder = None
    base_filename = None

    @staticmethod
    def init(custom_name=""):
        id = strftime("%Y-%m-%d_%Hh%Mmin%Ss", gmtime())
        FolderInfos.base_folder = "/".join(os.path.realpath(__file__).split("/")[:-3] + ["data/"]) + id + "_"+custom_name+"/"
        FolderInfos.base_filename = FolderInfos.base_folder + id+"_"+custom_name
        os.mkdir(FolderInfos.base_folder)