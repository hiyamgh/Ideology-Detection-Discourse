import os

years = [1983, 1984, 1989, 1990, 1991, 1992, 1993, 1994, 1995]

archives = ['An-Nahar', 'As-Safir']

with open('jobs_train_fasttext.txt', 'w') as f:
    for ar in archives:
        for y in years:
            train_file = f"/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/{y}-mod/training_file/{ar}/{y}_{ar}.txt"
            f.write(f"--archive {ar} --year {y} --train_file {train_file}\n")