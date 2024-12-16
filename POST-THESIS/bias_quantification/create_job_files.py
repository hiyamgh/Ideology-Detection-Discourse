# for weekly
with open('jobs_weekly.txt', 'w') as f:
    distances = ['cossim', 'norm', 'wasserstein']
    for model_name in ["aubmindlab-bert-base-arabertv2",
                       "aubmindlab-bert-base-arabertv01",
                       "aubmindlab-bert-base-arabert",
                       "aubmindlab-bert-base-arabertv02",
                       "aubmindlab-bert-base-arabertv02-twitter",
                       "UBC-NLP-ARBERT",
                       "UBC-NLP-MARBERT",
                       "qarib-bert-base-qarib",
                       "UBC-NLP-MARBERTv2"]:
        for dist in distances:
            f.write(f'--model_name {model_name} --split_by weekly --disttype {dist}\n')

# for biweekly
with open('jobs_biweekly.txt', 'w') as f:
    distances = ['cossim', 'norm', 'wasserstein']
    for model_name in ["aubmindlab-bert-base-arabertv2",
                       "aubmindlab-bert-base-arabertv01",
                       "aubmindlab-bert-base-arabert",
                       "aubmindlab-bert-base-arabertv02",
                       "aubmindlab-bert-base-arabertv02-twitter",
                       "UBC-NLP-ARBERT",
                       "UBC-NLP-MARBERT",
                       "qarib-bert-base-qarib",
                       "UBC-NLP-MARBERTv2"]:
        for dist in distances:
            f.write(f'--model_name {model_name} --split_by biweekly --disttype {dist}\n')

# for yearly
with open('jobs_yearly.txt', 'w') as f:
    distances = ['cossim', 'norm', 'wasserstein']
    for model_name in ["aubmindlab-bert-base-arabertv2",
                       "aubmindlab-bert-base-arabertv01",
                       "aubmindlab-bert-base-arabert",
                       "aubmindlab-bert-base-arabertv02",
                       "aubmindlab-bert-base-arabertv02-twitter",
                       "UBC-NLP-ARBERT",
                       "UBC-NLP-MARBERT",
                       "qarib-bert-base-qarib",
                       "UBC-NLP-MARBERTv2"]:
        for dist in distances:
            f.write(f'--model_name {model_name} --split_by yearly --disttype {dist}\n')

