with open("jobs_vocabulary.txt", "w") as f:
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

            f.write(f"--archive_name {archive} --model_name {model_name}\n")
f.close()