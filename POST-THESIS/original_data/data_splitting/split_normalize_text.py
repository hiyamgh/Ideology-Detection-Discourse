import sys
sys.path.append("..") # Adds higher directory to python modules path.

import os
import numpy as np
import sys
sys.path.insert(0, "/onyx/data/p118/POST-THESIS/original_data/")
from Arabic_Processing.normalization import *
from Arabic_Processing.stopword_removing import StopwordRemover
import argparse


def mkdir(folder_name):
    """ create a folder if it does not already exist, otherwise (if exists already), nothing will happen """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


if __name__ == '__main__':

    archive2dirs = {"nahar": "../nahar_transformed/", "assafir": "../assafir_transformed/"}

    stopword_remover = StopwordRemover()

    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument("--archive", type=str, default=None, help="name of the archive to split the text for")
    parser.add_argument("--month", type=int, default=None, help="month number to consider. If none, will assume a yearly split")
    parser.add_argument("--year", type=int, default=1982, help="Year number to consider. If month is None, will do a yearly split. Otherwise, will do a month-year split")
    args = parser.parse_args()

    normalizer = ArabicNormalizer() # pre-process arabic text (remove diacritics, unwanted chars, etc)

    delimiters = [EXCLAMATION, en_FULL_STOP, en_SEMICOLON, en_QUESTION, ar_FULL_STOP, ar_SEMICOLON, ar_QUESTION]

    sentence_lengths = []

    if args.month is not None:
        save_dir = 'txt_files/{}/monthly/'.format(args.archive)
        mkdir(folder_name=save_dir)

        with open(os.path.join(save_dir, "{}-{}.txt".format(args.year, args.month)), "w", encoding="utf-8") as f:
            rootdir = archive2dirs[args.archive]
            for file in os.listdir(rootdir):
                if os.path.isfile(os.path.join(rootdir, file)):
                    if file.startswith("._"):
                        continue
                    if file.endswith(".hocr"):
                        continue
                    if ".txt" not in file:
                        continue
                    if not file[0].isdigit():
                        continue

                    yearf = file[:2]
                    monthf = file[2:4]
                    dayf = file[4:6]
                    page_nbf = file[6:8]

                    if int(yearf) < 20:
                        year = int('20' + yearf)
                    else:
                        year = int('19' + yearf)

                    if monthf[0] == '0':
                        month = int(monthf[1])
                    else:
                        month = int(monthf)

                    # print(file, year, args.year, month, args.month)
                    if year == args.year and month == args.month:
                        with open(os.path.join(rootdir, file), "r", encoding="utf-8") as fin:
                            lines = fin.readlines()
                            for line in lines:
                                if line.strip() in ['\n', '']:
                                    continue
                                c_line = stopword_remover.remove_stopwords(
                                    sentence=line.strip())  # the line with Arabic stopwords removed, if any
                                c_line2 = normalizer.normalize_sentence(line=c_line)  # cleaned version of the line
                                if any([d in c_line2 for d in delimiters]):
                                    for d in delimiters:
                                        c_line2 = c_line2.replace(d, "")
                                if '\n' in c_line2:
                                    c_line2 = c_line2.replace("\n", "")
                                if c_line2.strip() == "":
                                    continue
                                f.write(c_line2.strip() + '\n')
                                sentence_lengths.append(len(c_line2.strip().split()))
                        fin.close()
        f.close()
    else:
        save_dir = 'txt_files/{}/yearly/'.format(args.archive)
        mkdir(folder_name=save_dir)
        with open(os.path.join(save_dir, "{}.txt".format(args.year)), "w", encoding="utf-8") as f:
            rootdir = archive2dirs[args.archive]
            for file in os.listdir(rootdir):
                if os.path.isfile(os.path.join(rootdir, file)):
                    if file.startswith("._"):
                        continue
                    if file.endswith(".hocr"):
                        continue
                    if ".txt" not in file:
                        continue
                    if not file[0].isdigit():
                        continue

                    yearf = file[:2]
                    monthf = file[2:4]
                    dayf = file[4:6]
                    page_nbf = file[6:8]

                    if int(yearf) < 20:
                        year = int('20' + yearf)
                    else:
                        year = int('19' + yearf)

                    if monthf[0] == '0':
                        month = int(monthf[1])
                    else:
                        month = int(monthf)

                    if year == args.year:
                        with open(os.path.join(rootdir, file), "r", encoding="utf-8") as fin:
                            lines = fin.readlines()
                            for line in lines:
                                if line.strip() in ['\n', '']:
                                    continue
                                c_line = stopword_remover.remove_stopwords(
                                    sentence=line.strip())  # the line with Arabic stopwords removed, if any
                                c_line2 = normalizer.normalize_sentence(line=c_line)  # cleaned version of the line
                                if any([d in c_line2 for d in delimiters]):
                                    for d in delimiters:
                                        c_line2 = c_line2.replace(d, "")
                                if '\n' in c_line2:
                                    c_line2 = c_line2.replace("\n", "")
                                if c_line2.strip() == "":
                                    continue
                                f.write(c_line2.strip() + '\n')
                                sentence_lengths.append(len(c_line2.strip().split()))
                        fin.close()
        f.close()
    print(f"average sentence length: {np.average(sentence_lengths)}")
    print(f"max length of a sentence: {np.max(sentence_lengths)}")
    print(f"min length of a sentence: {np.min(sentence_lengths)}")









