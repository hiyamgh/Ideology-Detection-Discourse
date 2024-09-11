import os
import shutil


def mkdir(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


if __name__ == '__main__':

    nahar_dir1 = 'nahar/nahar/nahar-batch-1/'
    nahar_dir2 = 'nahar/nahar/nahar-batch-2/'
    nahar_dir3 = 'nahar/nahar/nahar-batch-3/'
    nahar_dir4 = 'nahar/nahar/nahar-batch-4/'

    # assafir directories
    assafir_dir1 = 'assafir/assafir/assafir-batch-1/'
    assafir_dir2 = 'assafir/assafir/assafir-batch-2/'

    nahar_dirs = [nahar_dir1, nahar_dir2, nahar_dir3, nahar_dir4]  # all nahar directories in one list - helps in looping
    assafir_dirs = [assafir_dir1, assafir_dir2]  # all nahar directories in one list - helps in looping

    archive2dirs = {"nahar": nahar_dirs, "assafir": assafir_dirs}


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
                    if ".txt" not in file:
                        continue
                    if not file[:2].isdigit():
                        continue

                    yearf = file[:2]
                    monthf = file[2:4]
                    dayf = file[4:6]
                    page_nbf = file[6:8]

                    if int(yearf) < 20:
                        year = int('20' + yearf)
                    else:
                        # print(yearf, file)
                        year = int('19' + yearf)

                    if year == 1982 and page_nbf in ['01', '02'] and monthf in ['06', '07', '08', '09', '10', '11', '12']:
                        save_dir = 'opinionated_articles/{}/{}/'.format(archive, str(year))
                        mkdir(save_dir)
                        shutil.copyfile(os.path.join(subdir, file), os.path.join(save_dir, file))