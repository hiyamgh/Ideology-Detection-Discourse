# for weekly
with open('jobs_generate_embeddings_weekly.txt', 'w') as f:
    for archive in ["An-Nahar", "As-Safir"]:
        for model_name in ["aubmindlab-bert-base-arabertv2",
                           "aubmindlab-bert-base-arabertv01",
                           "aubmindlab-bert-base-arabert",
                           "aubmindlab-bert-base-arabertv02",
                           "aubmindlab-bert-base-arabertv02-twitter",
                           "UBC-NLP-ARBERT",
                           "UBC-NLP-MARBERT",
                           "qarib-bert-base-qarib",
                           "UBC-NLP-MARBERTv2"]:
            f.write(f'--archive {archive} --path_to_model /onyx/data/p118/POST-THESIS/generate_bert_embeddings/trained_models/{archive}/{model_name}/ --split_by weekly\n')


# for biweekly
with open('jobs_generate_embeddings_biweekly.txt', 'w') as f:
    for archive in ["An-Nahar", "As-Safir"]:
        for model_name in ["aubmindlab-bert-base-arabertv2",
                           "aubmindlab-bert-base-arabertv01",
                           "aubmindlab-bert-base-arabert",
                           "aubmindlab-bert-base-arabertv02",
                           "aubmindlab-bert-base-arabertv02-twitter",
                           "UBC-NLP-ARBERT",
                           "UBC-NLP-MARBERT",
                           "qarib-bert-base-qarib",
                           "UBC-NLP-MARBERTv2"]:
            f.write(f'--archive {archive} --path_to_model /onyx/data/p118/POST-THESIS/generate_bert_embeddings/trained_models/{archive}/{model_name}/ --split_by biweekly\n')


# for yearly
with open('jobs_generate_embeddings_yearly.txt', 'w') as f:
    for archive in ["An-Nahar", "As-Safir"]:
        for model_name in ["aubmindlab-bert-base-arabertv2",
                           "aubmindlab-bert-base-arabertv01",
                           "aubmindlab-bert-base-arabert",
                           "aubmindlab-bert-base-arabertv02",
                           "aubmindlab-bert-base-arabertv02-twitter",
                           "UBC-NLP-ARBERT",
                           "UBC-NLP-MARBERT",
                           "qarib-bert-base-qarib",
                           "UBC-NLP-MARBERTv2"]:
            f.write(f'--archive {archive} --path_to_model /onyx/data/p118/POST-THESIS/generate_bert_embeddings/trained_models/{archive}/{model_name}/ --split_by yearly\n')

