import os
import argparse


def mkdir(folder_name):
    """ create a folder if it does not already exist, otherwise (if exists already), nothing will happen """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


if __name__ == '__main__':
    # nahar directories
    nahar_dir1 = '../nahar/nahar/nahar-batch-1/'
    nahar_dir2 = '../nahar/nahar/nahar-batch-2/'
    nahar_dir3 = '../nahar/nahar/nahar-batch-3/'
    nahar_dir4 = '../nahar/nahar/nahar-batch-4/'

    # assafir directories
    assafir_dir1 = '../assafir/assafir/assafir-batch-1/'
    assafir_dir2 = '../assafir/assafir/assafir-batch-2/'

    nahar_dirs = [nahar_dir1, nahar_dir2, nahar_dir3,  nahar_dir4]  # all nahar directories in one list - helps in looping
    assafir_dirs = [assafir_dir1, assafir_dir2]  # all nahar directories in one list - helps in looping

    archive2dirs = {"nahar": nahar_dirs, "assafir": assafir_dirs}

    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument("--archive", type=str, default=None, help="name of the archive to joint the text for")
    args = parser.parse_args()

    save_dir = "../{}_transformed/".format(args.archive)
    mkdir(folder_name=save_dir)

    for dir in archive2dirs[args.archive]:
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

                with open(os.path.join(subdir, file), 'r', encoding='utf-8') as fin:
                    lines = fin.readlines()
                    transformed = [[]]
                    for line in lines:
                        if line == '\n':
                            transformed.append([])
                        else:
                            line = line.replace("\n", "")
                            transformed[-1].append(line)
                fin.close()
                with open(os.path.join(save_dir, file), 'w', encoding='utf-8') as fout:
                    for sublist in transformed:
                        if sublist == []:
                            continue
                        if " ".join(sublist).strip().replace("\n", "") == "":
                            continue
                        fout.write(" ".join(sublist) + "\n")
                fout.close()



    # with open('output.txt', 'w', encoding='utf-8') as fout:
    #     transformed = [[]]
    #     with open('../Arabic_Processing/04111314.txt', 'r', encoding='utf-8') as fin:
    #         lines = fin.readlines()
    #         for line in lines:
    #             if line == '\n':
    #                 transformed.append([])
    #             else:
    #                 line = line.replace("\n", "")
    #                 transformed[-1].append(line)
    #     fin.close()
    #     for sublist in transformed:
    #         if sublist == []:
    #             continue
    #         fout.write(" ".join(sublist) + "\n")
    #     fout.close()


#     # if i == 0:
#             #     idx_start = i
#             #     idx_end = -1
#             # if line == '\n':
#             #     idx_end = i - 1
#             #     sentence = " ".join(lines[idx_start: idx_end+1])
#             #     idx_start = i + 1



