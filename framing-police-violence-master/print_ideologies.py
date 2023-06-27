import pandas as pd

df = pd.read_csv('../semantic_shifts/wikipedia/datasets/poltical_parties.csv')
ideologies = list(set(list(df['Ideology'])))

c = 0
ideologies_clean = []
for ideo in ideologies:
    ideo = str(ideo)
    # if c % 10 != 0 or c == 0:

    if ideo.strip() == 'nan':
        continue
    if 'http' in ideo or 'htm' in ideo:
        continue
    if '\"' in ideo:
        continue

    if ',' in ideo:
        ideo_splitted = ideo.split(',')
        for ideo_s in ideo_splitted:
            if ':' not in ideo:
                if 'Historical' not in ideo:
                    if 'see below' not in ideo:
                        ideologies_clean.append(ideo_s)
                        c += 1
    else:

        # print(ideo, end=',')
    # else:
    #     print(ideo + ',')
        if ':' not in ideo:
            if 'Historical' not in ideo:
                if 'see below' not in ideo:
                    ideologies_clean.append(ideo)
                    c += 1

print('\ntotal count: {}'.format(len(list(set(ideologies_clean)))))

for ideo in ideologies_clean:
    if '-' in ideo and 'anti' in ideo.lower():
        ideologies_clean.append(ideo.split('-')[1].strip())

ideologies_clean = list(set(ideologies_clean))
print('updated total count: {}'.format(len(ideologies_clean)))

c = 0
for ideo in ideologies_clean:
    if c%10 != 0 or c == 0:
        print('\'{}\''.format(ideo), end=',')
    else:
        print('\'{}\''.format(ideo))
