'''
    * This code is responsible for gathering only the opinionated pages from an issue
    * We assume this is only pages 1 and 2 from a certain issue
    * The way Arabic Newspapers operated back then, was that there were certain articles that
    start on one page and continue on another page (not necessarily the page right after). We call it
    تتمة
    therefore, this code gets opinionated text from pages 1 and 2, and other completions
'''

import os
import pandas as pd
import shutil

months2num = {
    'January':      '01',
    'February':     '02',
    'March':        '03',
    'April':        '04',
    'May':          '05',
    'June':         '06',
    'July':         '07',
    'August':       '08',
    'September':    '09',
    'October':      '10',
    'November':     '11',
    'December':     '12'
}

monthcols2tatimacols = {
    'June': 'Unnamed: 1',
    'July': 'Unnamed: 3',
    'August': 'Unnamed: 5',
    'September': 'Unnamed: 7',
    'October': 'Unnamed: 9',
    'November': 'Unnamed: 11',
    'December': 'Unnamed: 13',
}


def mkdir(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


if __name__ == '__main__':

    nahar_dir1 = '/onyx/data/p118/POST-THESIS/original_data/nahar/nahar/nahar-batch-1/'
    nahar_dir2 = '/onyx/data/p118/POST-THESIS/original_data/nahar/nahar/nahar-batch-2/'
    nahar_dir3 = '/onyx/data/p118/POST-THESIS/original_data/nahar/nahar/nahar-batch-3/'
    nahar_dir4 = '/onyx/data/p118/POST-THESIS/original_data/nahar/nahar/nahar-batch-4/'

    # assafir directories
    assafir_dir1 = '/onyx/data/p118/POST-THESIS/original_data/assafir/assafir/assafir-batch-1/'
    assafir_dir2 = '/onyx/data/p118/POST-THESIS/original_data/assafir/assafir/assafir-batch-2/'

    nahar_dirs = [nahar_dir1, nahar_dir2, nahar_dir3,  nahar_dir4]  # all nahar directories in one list - helps in looping
    assafir_dirs = [assafir_dir1, assafir_dir2]  # all nahar directories in one list - helps in looping

    archive2dirs = {"nahar": nahar_dirs, "assafir": assafir_dirs}

    df = pd.read_excel('opinionated_articles_DrNabil/1982/tatimmas/tatimma.xlsx')
    cols = df.columns
    print(cols)

    filenames = []

    for month in months2num:
        if month in cols:
            print(f'PROCESSING MONTH: {month} ===================================================')
            tatimmacol = monthcols2tatimacols[month]

            days = list(df[month])
            tatimmas = list(df[tatimmacol])

            for day, page in zip(days, tatimmas):
                if str(day) in ["nan", " ", ""]:
                    continue

                day = str(int(day))

                print(f'\n{day}, {page}')
                day_f = '0' + day if int(day) < 10 else day

                filename1 = '82{}{}01.txt'.format(months2num[month], day_f)
                filename2 = '82{}{}02.txt'.format(months2num[month], day_f)

                filenames.append(filename1)
                filenames.append(filename2)

                print(f'added {filename1}')
                print(f'added {filename2}')

                page = str(page)
                if page in ["nan", " ", ""]:
                    continue

                if '.0' in page:
                    page = page.replace('.0', '')

                if ',' in page:
                    pages = [p.strip() for p in page.split(',')]
                    pages = [p for p in pages if p != ""]
                    pages = ['0' + p if int(p) < 10 else p for p in pages]

                    for pagenum in pages:
                        filename3 = '82{}{}{}.txt'.format(months2num[month], day_f, pagenum)
                        filenames.append(filename3)
                        print(f'added {filename3}')
                else:

                    pagenum = '0' + page if int(page) < 10 else page
                    filename3 = '82{}{}{}.txt'.format(months2num[month], day_f, pagenum)
                    filenames.append(filename3)
                    print(f'added {filename3}')

    save_dir = 'opinionated_articles_DrNabil/1982/txt_files/'
    mkdir(save_dir)

    for archive in archive2dirs:
        for dir in archive2dirs[archive]:
            rootdir = dir
            for subdir, dirs, files in os.walk(rootdir):
                print(subdir)
                for file in files:
                    if file in filenames:
                        shutil.copyfile(os.path.join(subdir, file), os.path.join(save_dir, file))

    print('finished copying files')
