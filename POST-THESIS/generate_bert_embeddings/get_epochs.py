'''
    * This code is responsible for gathering only the opinionated pages from an issue
    * We assume this is only pages 1 and 2 from a certain issue
    * The way Arabic Newspapers operated back then, was that there were certain articles that
    start on one page and continue on another page (not necessarily the page right after). We call it
    تتمة (tatimma)
    therefore, this code gets opinionated text from pages 1 and 2, and other completions
'''

import os
import pandas as pd
import shutil
import argparse


def mkdir(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument("--archive_name", type=str, default="An-Nahar", help="The name of the archive we are fine tuning over")
    parser.add_argument("--year", type=str, default="1982", help="the year if interest")
    args = parser.parse_args()

    if args.archive_name == "An-Nahar":
        dir1 = '/onyx/data/p118/POST-THESIS/original_data/nahar_transformed/'
    else:
        dir1 = '/onyx/data/p118/POST-THESIS/original_data/assafir_transformed/'

    epochs_needed = ['1983', '1984', '1989', '1990', '1991', '1992', '1993', '1994', '1995']
    dirs = [dir1]  # all nahar directories in one list - helps in looping
    year = args.year
    archive = args.archive_name

    save_dir = f'opinionated_articles_DrNabil/{year}/txt_files/{archive}'
    mkdir(save_dir)

    rootdir = dir1
    for subdir, dirs, files in os.walk(rootdir):
        print(subdir)
        for file in files:
            if file.startswith(f"{year[-2:]}"):
                shutil.copyfile(os.path.join(subdir, file), os.path.join(save_dir, file))

        print('finished copying files')

        # generate one file that is a collation of all files collated together
        save_dir_input = save_dir
        save_dir = f'opinionated_articles_DrNabil/{year}/training_file/{archive}'
        mkdir(save_dir)

        with open(os.path.join(save_dir, f'{year}_{archive}.txt'), 'w') as f:
            for file in os.listdir(save_dir_input):
                with open(os.path.join(save_dir_input, file), 'r') as fin:
                    lines = fin.readlines()
                    for line in lines:
                        f.write(line)
                fin.close()
        print('Done creating final training file!')
