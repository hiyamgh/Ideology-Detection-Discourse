import os

if __name__ == '__main__':
    nahar_dir1 = 'nahar-images/nahar-images/nahar-batch-1/'
    nahar_dir2 = 'nahar-images/nahar-images/nahar-batch-2/'
    nahar_dir3 = 'nahar-images/nahar-images/nahar-batch-3/'
    nahar_dir4 = 'nahar-images/nahar-images/nahar-batch-4/'
    nahar_dirs = [nahar_dir1, nahar_dir2, nahar_dir3, nahar_dir4]

    # assafir directories
    assafir_dir1 = 'assafir-images/assafir-images/assafir-batch-1/'
    assafir_dir2 = 'assafir-iamges/assafir-images/assafir-batch-2/'
    assafir_dirs = [assafir_dir1, assafir_dir2]

    # hayat directories
    hayat_dir1 = 'hayat-images/hayat-images/hayat-batch-1/'
    hayat_dir2 = 'hayat-images/hayat-images/hayat-batch-2/'
    hayat_dirs = [hayat_dir1, hayat_dir2]

    archive2dirs = {"nahar": nahar_dirs, "assafir": assafir_dirs, "hayat": hayat_dirs}

    for archive in archive2dirs:
        for dir in archive2dirs[archive]:
            rootdir = dir
            for subdir, dirs, files in os.walk(rootdir):
                print(subdir)
                for file in files:
                    if file.startswith("._"):
                        continue
                    if file.endswith(".hocr"):
                        continue
                    if not file[0].isdigit():
                        continue
                    if 'm' in file:
                        print(os.path.join(subdir, file))
                    else:
                        pass

