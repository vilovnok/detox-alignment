import os


def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass