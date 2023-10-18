import os


def create_dir(dir):
    dirs = dir.split("/")
    path = ""
    for i in range(len(dirs)):
        path = path + dirs[i] + "/"
        if not os.path.exists(path):
            os.mkdir(path)
