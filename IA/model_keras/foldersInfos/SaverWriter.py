import json


class SaveWriter:

    def __init__(self, file_path):
        self.file_path = file_path
        self.dico = {}

    def __enter__(self):
        # ttysetattr etc goes here before opening and returning the file object
        self.fd = open(self.file_path, "r+")
        self.dico = json.load(self.fd)
        return self.dico

    def __exit__(self, type, value, traceback):
        # Exception handling here
        if value is None:
            self.fd.seek(0)
            json.dump(self.dico, self.fd,indent=4)
        self.fd.close()