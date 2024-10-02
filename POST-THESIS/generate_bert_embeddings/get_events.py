'''
    This code is for getting sentences belonging to events based on keyword search
    * we loop over the "opinionated articles" only
    * every event is associated with keywords, start date, end date
    * for every event, we get all sentences in between start_date and end_date, containing the keywords
'''

import pandas as pd
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta


def mkdir(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


events2info = {
    '1982_Lebanon_war': {
        'date': '820606', # date resembles the way the newspaper files are named
        'keywords': [
            'اجتياح اسرائيل',
            'قصف اسرائيل',
            'غارات اسرائيل',
            'هجمات اسرائيل',
            'حصار بيروت',
            'طيران اسرائيل',
        ],
        'window': '1 month'
    },

    '1982_Philip_Habib': {
        'date': '820607', # date resembles the way the newspaper files are named
        'keywords': [
            'فيليب حبيب',
            'مبعوث اميركي',
            'مبعوث امريكي',
            'وسيط اميركي',
            'وسيط امريكي',
            'حبيب اميركي'
        ],
        'window': '2 weeks'
    },

    'Alexander_Hague_resignation': {
        'date': '820625', # date resembles the way the newspaper files are named
        'keywords': [
            'استقال الكسندر هيغ',
            'استقال وزير خارجية اميركي',
            'استقال وزير خارجية امريكي',
        ],
        'window': '2 weeks'
    },

    'Bashir_Gemayel_elected_president': {
        'date': '820823', # date resembles the way the newspaper files are named
        'keywords': [
            'انتخاب شيخ بشير الجميل',
            'انتخاب بشير الجميل',
            'انتخاب رئيس'
        ],
        'window': '2 weeks'
    },

    'Arafat_exits_Beirut': {
        'date': '820830', # date resembles the way the newspaper files are named
        'keywords': [
            'رحيل ياسر عرفات',
            'مغادر ياسر عرفات',
            'انسحاب ياسر عرفات',
            'رحيل ابو عمار',
            'مغادر ابو عمار',
            'انسحاب ابو عمار'
        ],
        'window': '2 weeks'
    },

}


if __name__ == '__main__':
    archives = ["An-Nahar", "As-Safir"]
    for archive in archives:

        rootdir = f'opinionated_articles_DrNabil/1982/txt_files/{archive}'

        for event in events2info:

            start_date = datetime.strptime(events2info[event]['date'], '%y%m%d')
            window = events2info[event]['window']
            if 'month' in window:
                end_date = relativedelta(months=int(window.strip().split(' ')[0])) + start_date
            else:
                end_date = relativedelta(weeks=int(window.strip().split(' ')[0])) + start_date

            save_dir = f"opinionated_articles_DrNabil/1982/sentences_per_event/{event}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
            mkdir(save_dir)

            # get all files in between start date and end date
            filenames = []
            for subdir, dirs, files in os.walk(rootdir):
                print(subdir)
                for file in files:
                    file_date_only = file.split('.')[0][:-2]
                    date_actual = datetime.strptime(file_date_only, '%y%m%d')
                    if start_date <= date_actual <= end_date:
                        filenames.append(os.path.join(subdir, file))


            # get all lines containing any of the keywords
            data = {'lines': [], 'keywords': [], 'files': []}
            for file in filenames:
                with open(file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        keywords_found = ''
                        for keywords in events2info[event]['keywords']:
                            if all([k in line for k in keywords.strip().split(' ')]) or all([k in s for s in line.strip().split(' ') for k in keywords.strip().split(' ')]):
                                print()
                                if keywords_found == '':
                                    keywords_found += keywords
                                else:
                                    keywords_found += ', ' + keywords
                        if keywords_found != '':
                            data['lines'].append(line)
                            data['keywords'].append(keywords_found)
                            data['files'].append(file)

            # save data to excel sheet
            df = pd.DataFrame(data)
            df.to_excel(os.path.join(save_dir, f'{archive}.xlsx'), index=False)

