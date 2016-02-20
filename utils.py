import os


# Returns a list of paths to sub folders
def list_dirs(folder):
    return [d for d in (os.path.join(folder, d1) for d1 in os.listdir(folder))
            if os.path.isdir(d)]


# Clears the screen
def cls():
    os.system(['clear', 'cls'][os.name == 'nt'])
