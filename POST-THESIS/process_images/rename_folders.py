import shutil
import os


rootdir = './'
for subdir, dirs, files in os.walk(rootdir):
    if subdir == './':
        continue
    os.rename(subdir, subdir.strip().rstrip())