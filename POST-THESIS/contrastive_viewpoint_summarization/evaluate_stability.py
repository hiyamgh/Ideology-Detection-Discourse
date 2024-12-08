from itertools import cycle
from bidi import algorithm as bidialg
import arabic_reshaper
from words_are_malleable_contextualized import *
from scipy.signal import find_peaks
from scipy import stats
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

################### common parameters for all images ###################
plt.rcParams['figure.dpi'] = 300
################### end of common parameters         ###################

import seaborn as sns
import pandas as pd
import csv
import time
import argparse
from datetime import date


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def jaccard_similarity(listoflists):
    inter = set.intersection(*map(set, listoflists))
    un = set().union(*listoflists)
    return float(len(inter) / len(un))


# ax[0].set_xlabel("Combined" if i == 0 else "Neighbor-based" if i == 1 else "Linear-Mapping")


def get_contrastive_viewpoint_summary(w, summary_length, k,
                                      models,
                                      embeddings,
                                      tokenizers,
                                      models_names,
                                      dir_name_matrices,
                                      year,
                                      annoy_indexes,
                                      vocab_names,
                                      thresh=0.5):
    """ get a contrastive viewpoint summary of a word of length n. For a certain
        word:
        1. we get its top k nearest neighbors.
        2. Then for each nearest neighbor, we add it into the summary if its stability is equal to or less than a certain threshold.
    """
    summary1, summary2 = [], []
    all_summaries = []

    nns1 = [n for n in get_nearest_neighbors(word=w, year=year, embeddings=embeddings[0], tokenizer=tokenizers[0], model=models[0], annoy_index=annoy_indexes[0], vocab_names=vocab_names[0], K=k)]
    nns2 = [n for n in get_nearest_neighbors(word=w, year=year, embeddings=embeddings[1], tokenizer=tokenizers[1], model=models[1], annoy_index=annoy_indexes[1], vocab_names=vocab_names[1], K=k)]

    count = 0
    for nn in nns1:
        if count == summary_length:
            break

        st = get_stability_combined_one_word(models, embeddings, tokenizers,
                               models_names, annoy_indexes, vocab_names, dir_name_matrices, nn, year, k)

        if st <= thresh:
            summary1.append((st, nn))
            count += 1
    all_summaries.append(summary1)

    count = 0
    for nn in nns2:
        if count == summary_length:
            break

        st = get_stability_combined_one_word(models, embeddings, tokenizers,
                                             models_names, annoy_indexes, vocab_names, dir_name_matrices, nn, year, k)

        if st <= thresh:
            summary2.append((st, nn))
            count += 1
    all_summaries.append(summary2)

    print(f'summary of {w} from Nahar point of view: {summary1}')
    print(f'summary of {w} from AsSafir point of view: {summary2}')
    return all_summaries


def save_summary(summary2save, save_dir, category_name, word, word_name):
    mkdir(save_dir)
    with open(os.path.join(save_dir, f'{category_name}_{word_name}.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Summary for {word} from An-Nahar\n')
        for w in summary2save[0]:
            f.write(w + '\n')
        print('===============================================================')
        f.write(f'Summary for {word} from As-Safir\n')
        for w in summary2save[1]:
            f.write(w + '\n')
    f.close()

    # with open(os.path.join(save_dir, 'all_summaries.pickle'), 'wb') as handle:
    # with open(os.path.join(save_dir, 'all_summaries_monthly_political_parties.pickle'), 'wb') as handle:
    #     pickle.dump(summary2save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print(f'saved all_summaries dict into {save_dir}')


def get_week_coverage_sorted(archive):
    """
    loops over the .txt files and returns a sorted list of months-weeknbs
    """
    months_weeks = set()
    root_dir = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/txt_files/{}/'.format(archive)
    for file in os.listdir(root_dir):
        month = file[2:4]
        day = file[4:6]
        year = "1982"
        print(f'processing file {file} with year: {year}, month: {month}, day: {day}')
        given_date = date(int(year), int(month), int(day))
        week_number = given_date.isocalendar()[1]

        month_week = f"{month}_{str(week_number)}"

        months_weeks.add(month_week)

    months_weeks_l = list(months_weeks)
    months_weeks_ls = sorted(months_weeks_l, key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))
    print(f"months_weeks sorted for {archive}: {months_weeks_ls}")

    return months_weeks_ls


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="aubmindlab-bert-base-arabertv2", help="model name for archives")

    # neighbor and combined approach (words_file is needed by linear approach as well)
    parser.add_argument("--k", default=100, help="number of nearest neighbors to consider per word - for neighbours and combined approach")
    parser.add_argument("--threshold", default="0.5", help="threshold value(s) for generating contrastive viewpoint summaries")
    parser.add_argument("--cvs_len", default=20, help="length of the contrastive viewpoint summary")  # cvs ==> contrastive viewpoint summary
    parser.add_argument("--split_by", default="monthly", help="the level at which the summaries are generated -- `weekly` or `monthly`")
    parser.add_argument("--year", default="06", help="the year for which to create the summaries for")
    parser.add_argument("--category", default="political_parties", help="The name of the category as per the contasrtive summary enities file names")
    args = parser.parse_args()

    model_name = args.model_name

    path_nahar = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/embeddings/An-Nahar/{}/monthly/'.format(model_name)
    path_assafir = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/embeddings/As-Safir/{}/monthly'.format(model_name)

    path_to_model_nahar = "/onyx/data/p118/POST-THESIS/generate_bert_embeddings/trained_models/An-Nahar/{}/".format(model_name)
    path_to_model_assafir = "/onyx/data/p118/POST-THESIS/generate_bert_embeddings/trained_models/As-Safir/{}/".format(model_name)

    # tokenizers for each archive
    tokenizer_nahar = AutoTokenizer.from_pretrained(path_to_model_nahar)
    tokenizer_assafir = AutoTokenizer.from_pretrained(path_to_model_assafir)

    # models for each archive
    model_nahar = AutoModelForMaskedLM.from_pretrained(path_to_model_nahar, output_hidden_states=True).cuda()
    model_nahar.eval()

    model_assafir = AutoModelForMaskedLM.from_pretrained(path_to_model_assafir, output_hidden_states=True).cuda()
    model_assafir.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(path_nahar, 'embeddings.pickle'), 'rb') as handle:
        embeddings_nahar = pickle.load(handle)
        print(len(embeddings_nahar))

    with open(os.path.join(path_assafir, 'embeddings.pickle'), 'rb') as handle:
        embeddings_assafir = pickle.load(handle)
        print(len(embeddings_assafir))

    year = args.year
    category = args.category
    split_by = args.split_by
    thresh = args.threshold
    k = int(args.k)  # number of nearest neighbors to include when creating contrastive viewpoint summaries
    cvs_len = int(args.cvs_len)  # length of the contrastive viewpoint summary
    save_dir = f"contrastive_summaries/{model_name}-{split_by}/"
    mkdir(save_dir_matrices)

    rootdir = "../generate_bert_embeddings/entities/contrastive_summaries/"
    category_words = []
    for file in os.listdir(rootdir):
        category_name = file.replace(".txt", "")
        if category_name == category:
            with open(os.path.join(rootdir, file), "r") as f:
                lines = f.readlines()
                for line in lines:
                    w = line.strip().replace("\n", "")
                    category_words.append(w)
            f.close()
            break
    t1 = time.time()

    annoy_index_nahar, vocab_names_nahar = load_vocab_embeddings(year=year,  vocab_path=f"vocabulary_embeddings/An-Nahar/{model_name}-{split_by}/")  # the vocab path to a model fine tuned on An-Nahar
    annoy_index_assafir, vocab_names_assafir = load_vocab_embeddings(year=year, vocab_path=f"vocabulary_embeddings/As-Safir/{model_name}-{split_by}/")  # the vocab_path to a model fine tuned on As-Safir

    for i, w in enumerate(category_words):
        print('---- word: {} - timepoint: {} ----'.format(w, year))
        print('threshold: {}'.format(thresh))

        save_dir_matrices = f"transformation_matrices/{model_name}-monthly/"

        model1_name = f'1982-nahar-{year}'
        model2_name = f'1982-assafir-{year}'
        model_names = [model1_name, model2_name]

        # update summary
        summaries = get_contrastive_viewpoint_summary(w, summary_length=cvs_len,
                                                      k=k,
                                                      models=[model_nahar, model_assafir],
                                                      embeddings=[embeddings_nahar, embeddings_assafir],
                                                      tokenizers=[tokenizer_nahar, tokenizer_assafir],
                                                      models_names=[model1_name, model2_name],
                                                      dir_name_matrices=save_dir_matrices,
                                                      year=year,
                                                      annoy_indexes=[annoy_index_nahar, annoy_index_assafir],
                                                      vocab_names=[vocab_names_nahar, vocab_names_assafir],
                                                      thresh=thresh)

        save_summary(summary2save=summaries, save_dir=save_dir, category_name=category, word=w, word_name=f'word_{i}')

        t2 = time.time()
        print(f'time taken to complete 1 summary: {(t2 - t1) / 60} mins')

        # years_nahar = get_week_coverage_sorted(archive="An-Nahar")
        # years_assafir = get_week_coverage_sorted(archive="As-Safir")
        # combined_years = set(years_nahar).intersection(set(years_assafir))

        # combined_years = ['06', '07', '08', '09', '10', '11', '12']




