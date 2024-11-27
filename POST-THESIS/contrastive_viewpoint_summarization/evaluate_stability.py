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


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def filter_stability_neighbors(stability_neigh, stability_comb):
    """ filter out the unwanted neighbors of words from the neighbors-based approach i.e.
        keep only the words in the common vocabulary
    """
    stability_neigh_filtered = {}
    for k in stability_comb:
        stability_neigh_filtered[k] = stability_neigh[k]
    return stability_neigh_filtered


def print_items(anydict):
    for k, v in anydict.items():
        print(k, v)


def save_heads_tails_all(stabilities_comb, stabilities_neigh, stabilities_lin, n=25, verbose=True,
                         save_heads_tails=True,
                         save_dir=None,
                         file_name=None):
    """ gets the top n most unstable words, and the top n most stable words """
    # sort the stabilities dictionary by increasing order of stabilities (items at the beginning
    # have low stability - items at the end have high stability)
    stabilities_comb = {k: v for k, v in sorted(stabilities_comb.items(), key=lambda item: item[1])}
    stabilities_neigh = {k: v for k, v in sorted(stabilities_neigh.items(), key=lambda item: item[1])}
    stabilities_lin = {k: v for k, v in sorted(stabilities_lin.items(), key=lambda item: item[1])}

    iterations = [(stabilities_comb, 'Combination'), (stabilities_neigh, 'Neighbor-based'),
                  (stabilities_lin, 'Linear-Mapping')
                  ]

    heads_all = {}
    tails_all = {}

    for item in iterations:
        stabilities, name = item[0], item[1]
        # first n items = heads = the most unstable words
        heads = {k: stabilities[k] for k in list(stabilities)[:n]}
        # last n items = tails = the most stable words
        tails = {k: stabilities[k] for k in list(stabilities)[-n:]}

        heads_mod = [(k, v) for k, v in heads.items()]
        tails_mod = [(k, v) for k, v in tails.items()]

        if verbose:
            print('{} - heads:'.format(name))
            print_items(heads)
            print('{} - tails:'.format(name))
            print_items(tails)

        heads_all[name] = heads
        tails_all[name] = tails

        if save_heads_tails:
            mkdir(save_dir)
            with open(os.path.join(save_dir, '{}_{}.csv'.format(file_name, name)), 'w', encoding='utf-8-sig',
                      newline='') as f:
                r = csv.writer(f)
                r.writerow(['heads', 'tails'])

                for i in range(len(heads_mod)):
                    r.writerow(
                        [heads_mod[i][0] + "," + str(heads_mod[i][1]), tails_mod[i][0] + "," + str(tails_mod[i][1])])

    return heads_all, tails_all


def get_heads_tails(stabilities, n, verbose=True):
    """ gets the top n most unstable words, and the top n most stable words """
    # sort the stabilities dictionary by increasing order of stabilities (items at the beginning
    # have low stability - items at the end have high stability)
    stabilities = {k: v for k, v in sorted(stabilities.items(), key=lambda item: item[1])}

    # first n items = heads = the most unstable words
    heads = {k: stabilities[k] for k in list(stabilities)[:n]}
    # last n items = tails = the most stable words
    tails = {k: stabilities[k] for k in list(stabilities)[-n:]}
    if verbose:
        print('heads:')
        print_items(heads)
        print('tails:')
        print_items(tails)
    return heads, tails


def get_stability_words(stabilities, words):
    """ prints the stability value of each word """
    for w in words:
        if w in stabilities:
            print('{}: {}'.format(w, str(stabilities[w])))
        else:
            print('word {} not found in teh dictionary'.format(w))


def jaccard_similarity(listoflists):
    inter = set.intersection(*map(set, listoflists))
    un = set().union(*listoflists)
    return float(len(inter) / len(un))


# ax[0].set_xlabel("Combined" if i == 0 else "Neighbor-based" if i == 1 else "Linear-Mapping")


def generate_stability_heatmap(words, stability_dicts_combined, stability_dicts_neighbor,
                               stability_dicts_linear,
                               years, save_dir, fig_name):
    yticks = [bidialg.get_display(arabic_reshaper.reshape(w)) for w in words]
    numxticks = len(stability_dicts_combined)
    fig, ax = plt.subplots(nrows=1, ncols=3)
    data_comb, data_neigh, data_lin = [], [], []
    for w in words:
        stab_vals_comb, stab_vals_neigh, stab_vals_lin = [], [], []
        for j in range(len(years)):
            stab_vals_comb.append(stability_dicts_combined[j][w])
            stab_vals_neigh.append(stability_dicts_neighbor[j][w])
            stab_vals_lin.append(stability_dicts_linear[j][w])
        data_comb.append(stab_vals_comb)
        data_neigh.append(stab_vals_neigh)
        data_lin.append(stab_vals_lin)

    data_comb = np.array(data_comb)
    sns.heatmap(data_comb, vmin=-0.1, vmax=1.0, yticklabels=yticks, cmap="YlGnBu", cbar=False, ax=ax[0])
    ax[0].set_xlabel("Combined")
    ax[0].set_xticks(list(range(numxticks)))
    ax[0].set_xticklabels(years, rotation=90)

    data_neigh = np.array(data_neigh)
    sns.heatmap(data_neigh, vmin=-0.1, vmax=1.0, yticklabels=yticks, cmap="YlGnBu", cbar=False, ax=ax[1])
    ax[1].set_xlabel("Neighbor-based")
    ax[1].set_xticks(list(range(numxticks)))
    ax[1].set_xticklabels(years, rotation=90)

    data_lin = np.array(data_lin)
    sns.heatmap(data_lin, vmin=-0.1, vmax=1.0, yticklabels=yticks, cmap="YlGnBu", cbar=False, ax=ax[2])
    ax[2].set_xticks(list(range(numxticks)))
    ax[2].set_xlabel("Linear-Mapping")
    ax[2].set_xticklabels(years, rotation=90)

    mkdir(save_dir)
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, fig_name + '_stabilities_heatmap.png'))
    plt.close()


def plot_jaccard_similarity_tails(stability_dicts_combined, stability_dicts_neighbor, stability_dicts_linear, n_sizes,
                                  save_dir, fig_name):
    """ get the jaccard similarity between the tails of the different stability
        approaches for different sizes of n. Ideally, because words in tails should be stable,
        they must be present in the tails of any corpus used.
    """
    jaccard_sims_comb, jaccard_sims_neigh, jaccard_sims_lin = [], [], []
    for n in n_sizes:
        # get the jaccard similarity over the "combined"-based dictionaries
        all_tails = []
        for stab_dict in stability_dicts_combined:
            _, tails = get_heads_tails(stab_dict, n, verbose=False)
            all_tails.append(tails)
        jaccard = jaccard_similarity(all_tails)
        jaccard_sims_comb.append(jaccard)

        # get the jaccard similarity over the "neighbor"-based dictionaries
        all_tails = []
        for stab_dict in stability_dicts_neighbor:
            _, tails = get_heads_tails(stab_dict, n, verbose=False)
            all_tails.append(tails)
        jaccard = jaccard_similarity(all_tails)
        jaccard_sims_neigh.append(jaccard)

        # get the jaccard similarity over the "linear"-based dictionaries
        all_tails = []
        for stab_dict in stability_dicts_linear:
            _, tails = get_heads_tails(stab_dict, n, verbose=False)
            all_tails.append(tails)
        jaccard = jaccard_similarity(all_tails)
        jaccard_sims_lin.append(jaccard)

    lines = ["--", "-.", ":"]
    linecycler = cycle(lines)
    plt.figure()
    for i in range(len(lines)):
        if i == 0:
            plt.plot(list(n_sizes), jaccard_sims_comb, next(linecycler), label="Combination")
        elif i == 1:
            plt.plot(list(n_sizes), jaccard_sims_neigh, next(linecycler), label="Neighbors-based")
        else:
            plt.plot(list(n_sizes), jaccard_sims_lin, next(linecycler), label="Linear-Mapping")
    plt.legend()
    plt.xlabel('tail sizes')
    plt.ylabel('jaccard similarity')
    plt.xlim([n_sizes[0], n_sizes[-1]])
    # plt.ylim([0, max(max(jaccard_sims_comb), max(jaccard_sims_neigh), max(jaccard_sims_lin))])
    plt.ylim([0, 0.1])
    mkdir(save_dir)
    plt.savefig(os.path.join(save_dir, fig_name + '_jaccard_similarities.png'))
    plt.close()


def plot_delta_ranks_words(ranks_comb, ranks_neigh, ranks_lin, words,
                           save_dir, fig_name):
    deltas_neigh, deltas_lin = [], []
    words_decoded = []
    for w in words:
        drneigh = ranks_neigh[w] - ranks_comb[w]  # get the delta rank
        drlin = ranks_lin[w] - ranks_comb[w]  # get the delta rank
        deltas_neigh.append(drneigh)  # add to list
        deltas_lin.append(drlin)  # add to list
        words_decoded.append(
            bidialg.get_display(arabic_reshaper.reshape(w)))  # decode arabic word to make it appear in matplotlib
    plt.bar(words_decoded, deltas_neigh, label='Combination vs. Neighbor-based')
    plt.bar(words_decoded, deltas_lin, label='Combination vs. Linear Mapping', bottom=deltas_neigh)
    plt.xticks(rotation=90)
    plt.ylabel(r'$\Delta$' + 'rank')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    fig.tight_layout()
    mkdir(save_dir)
    plt.savefig(os.path.join(save_dir, fig_name + '_delta_ranks.png'))
    plt.close()


def get_ranks(stability_combined, stability_neighbors, stability_linear):
    # sort the stabilities dictionary by increasing order of stabilities (items at the beginning
    # have low stability - items at the end have high stability)
    stability_combined = {k: v for k, v in sorted(stability_combined.items(), key=lambda item: item[1])}
    stability_neighbors = {k: v for k, v in sorted(stability_neighbors.items(), key=lambda item: item[1])}
    stability_linear = {k: v for k, v in sorted(stability_linear.items(), key=lambda item: item[1])}

    values_combined = list(stability_combined.values())
    values_neighbor = list(stability_neighbors.values())
    values_linear = list(stability_linear.values())

    ranks_combined, ranks_neigh, ranks_lin = [], [], []

    ranks_combined.append(1)  # for the first value, its rank is 1
    ranks_neigh.append(1)  # for the first value, its rank is 1
    ranks_lin.append(1)

    # get the rankings per value for the combined
    rank = 1
    for i in range(1, len(values_combined[1:]) + 1):
        if round(values_combined[i], 5) == round(values_combined[i - 1], 5):
            ranks_combined.append(rank)
        else:
            rank += 1
            ranks_combined.append(rank)
    print(len(ranks_combined) == len(values_combined))

    # get the rankings per value for the neighbors
    rank = 1
    for i in range(1, len(values_neighbor[1:]) + 1):
        if round(values_neighbor[i], 8) == round(values_neighbor[i - 1], 8):
            ranks_neigh.append(rank)
        else:
            rank += 1
            ranks_neigh.append(rank)
    print(len(ranks_neigh) == len(values_neighbor))

    # get the rankings per value for the linear
    rank = 1
    for i in range(1, len(values_linear[1:]) + 1):
        if round(values_linear[i], 8) == round(values_linear[i - 1], 8):
            ranks_lin.append(rank)
        else:
            rank += 1
            ranks_lin.append(rank)
    print(len(ranks_lin) == len(values_linear))

    ranks_combined = dict(zip(list(stability_combined.keys()), ranks_combined))
    ranks_neigh = dict(zip(list(stability_neighbors.keys()), ranks_neigh))
    ranks_lin = dict(zip(list(stability_linear.keys()), ranks_lin))

    return ranks_combined, ranks_neigh, ranks_lin


def get_contrastive_viewpoint_summary(w, summary_length, k,
                                      models,
                                      embeddings,
                                      tokenizers,
                                      models_names,
                                      mat_name,
                                      year,
                                      thresh=0.5):
    """ get a contrastive viewpoint summary of a word of length n. For a certain
        word:
        1. we get its top k nearest neighbors.
        2. Then for each nearest neighbor, we add it into the summary if its stability is equal to or less than a certain threshold.
    """
    summary1, summary2 = [], []
    all_summaries = []

    nns1 = [n for n in get_nearest_neighbors(word=w, year=year, embeddings=embeddings[0], tokenizer=tokenizers[0], model=models[0])]
    nns2 = [n for n in get_nearest_neighbors(word=w, year=year, embeddings=embeddings[1], tokenizer=tokenizers[1], model=models[1])]

    count = 0
    for nn in nns1:
        if count == summary_length:
            break

        st = get_stability_combined_one_word(models, embeddings, tokenizers,
                               models_names, mat_name, nn, year, k)

        if st <= thresh:
            summary1.append((st, nn))
            count += 1
    all_summaries.append(summary1)

    count = 0
    for nn in nns2:
        if count == summary_length:
            break

        st = get_stability_combined_one_word(models, embeddings, tokenizers,
                                             models_names, mat_name, nn, year, k)

        if st <= thresh:
            summary2.append((st, nn))
            count += 1
    all_summaries.append(summary2)

    print(f'summary of {w} from Nahar point of view: {summary1}')
    print(f'summary of {w} from AsSafir point of view: {summary2}')
    return all_summaries


def perform_paired_t_test(ranks_comb, ranks_neigh, ranks_lin, save_dir, file_name):
    # get test-statistic and p-value results
    result_comb_neigh = stats.ttest_rel(list(ranks_comb.values()), list(ranks_neigh.values()))
    result_comb_lin = stats.ttest_rel(list(ranks_comb.values()), list(ranks_lin.values()))

    # get the average rank of each method
    avg_rank_comb = np.mean(list(ranks_comb.values()))
    avg_rank_neigh = np.mean(list(ranks_neigh.values()))
    avg_rank_lin = np.mean(list(ranks_lin.values()))

    print('avg rank combined: {}'.format(np.mean(list(ranks_comb.values()))))
    print('avg rank neighbors: {}'.format(np.mean(list(ranks_neigh.values()))))
    print('avg rank linear: {}'.format(np.mean(list(ranks_lin.values()))))

    df = pd.DataFrame(columns=['Avg. Rank Comb', 'Avg. Rank Neigh', 'Avg. Rank Lin',
                               'ttest_comb_neigh', 'ttest_comb_lin',
                               'pval_comb_neigh', 'pval_comb_lin'])

    df = df.append({
        'Avg. Rank Comb': avg_rank_comb,
        'Avg. Rank Neigh': avg_rank_neigh,
        'Avg. Rank Lin': avg_rank_lin,
        'ttest_comb_neigh': result_comb_neigh[0],
        'ttest_comb_lin': result_comb_lin[0],
        'pval_comb_neigh': result_comb_neigh[1],
        'pval_comb_lin': result_comb_lin[1]
    }, ignore_index=True)

    if result_comb_neigh[1] < 0.05:
        print(result_comb_neigh)
        print('accept H1 for comb-neigh')
    else:
        print(result_comb_neigh)
        print('accept H0 for comb-neigh')

    if result_comb_lin[1] < 0.05:
        print(result_comb_lin)
        print('accept H1 for comb-lin')
    else:
        print(result_comb_lin)
        print('accept H0 for comb-lin')

    mkdir(save_dir)
    df.to_csv(os.path.join(save_dir, file_name + '.csv'), index=False)


def read_keywords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        words = f.readlines()
    words = [w[:-1] for w in words if '\n' in w]
    words = [w for w in words if w.strip() != '']
    words = [w.strip() for w in words]
    return words


def save_summary(summary2save, save_dir):
    mkdir(save_dir)
    with open(os.path.join(save_dir, 'all_summaries.pickle'), 'wb') as handle:
        pickle.dump(summary2save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'saved all_summaries dict into {save_dir}')


def plot_stabilities_over_time_heatmap(words_batches, stabilities_over_time, mode, save_dir, batch_names, fig_name):
    temp = np.empty([len(words_batches), len(stabilities_over_time)])
    stabilities = np.zeros_like(temp)

    for i, batch in enumerate(words_batches):
        for w in batch:
            for j, tp in enumerate(stabilities_over_time):
                st = stabilities_over_time[tp][w]
                stabilities[i][j] += st

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(stabilities, linewidths=.5, yticklabels=batch_names)
    ax.set_xticklabels(list(stabilities_over_time.keys()))

    type_analysis = 'diachronic' if mode[:2] == 'd-' else 'synchronic'
    archive = mode[2:] if mode != 's' else None
    if archive is not None:
        xlab = 'time points of the stability in {} archive - {} analysis'.format(archive, type_analysis)
    else:
        xlab = 'time points of the stability - {} analysis'.format(type_analysis)
    plt.xticks(rotation=45)
    plt.xlabel(xlab)
    plt.ylabel('stability')
    plt.tight_layout()
    mkdir(save_dir)
    plt.savefig(os.path.join(save_dir, fig_name + '.png'))
    plt.savefig(os.path.join(save_dir, fig_name + '.pdf'))
    plt.close()


def get_stability_statistics_over_time(words_batches, stabilities_over_time, save_dir):
    ''' Get max, min, and average stability across all words-times and per word '''
    all_words = []
    for batch in words_batches:
        for word in batch:
            all_words.append(word)

    averages = []
    average_word = {}
    for time_point in stabilities_over_time:
        for word in stabilities_over_time[time_point]:
            if word in all_words:
                val = stabilities_over_time[time_point][word]
                averages.append(val)
                if word in average_word:
                    average_word[word].append(val)
                else:
                    average_word[word] = [val]
    total_average = np.sum(np.array(averages)) / len(averages)
    print('total average stability (across all words; across all time points): {}'.format(total_average))
    print('max stability attained (across all words; across all time points): {}'.format(np.max(np.array(averages))))
    print('min stability attained (across all words; across all time points): {}'.format(np.min(np.array(averages))))
    print('saving average stability per word (across all time points):')

    mkdir(save_dir)
    with open(os.path.join(save_dir, 'average_stability.csv'), 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['word', 'avg. stability', 'max stability', 'min stability']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for w in average_word:
            # print('word: {} - average stability: {}'.format(w, np.sum(average_word[w]) / len(average_word[w])))
            writer.writerow({'word': '{}'.format(w),
                             'avg. stability': '{}'.format(np.sum(average_word[w]) / len(average_word[w])),
                             'max stability': '{}'.format(np.max(np.array(average_word[w]))),
                             'min stability': '{}'.format(np.min(np.array(average_word[w])))
                             })


def plot_stabilities_over_time_lineplot(words_batches, stabilities_over_time, mode, save_dir, batch_names, fig_name):
    # https://stats.stackexchange.com/questions/118033/best-series-of-colors-to-use-for-differentiating-series-in-publication-quality
    # https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=8
    all_mins, all_maxs = [], []
    for i, batch in enumerate(words_batches):
        stabilities = [0 for _ in range(len(stabilities_over_time))]
        for w in batch:
            stabilities_temp = []
            for tp in stabilities_over_time:  # tp meaning time_point
                st = stabilities_over_time[tp][w]
                stabilities_temp.append(st)
            stabilities = [orig + curr for orig, curr in zip(stabilities, stabilities_temp)]
        # w_proc = bidialg.get_display(arabic_reshaper.reshape(w))

        # find peaks and troughs
        peaks, _ = find_peaks(np.array(stabilities))
        mins, _ = find_peaks(np.array(stabilities) * -1)

        # add the peaks and mins to an array
        all_maxs.extend(peaks)
        all_mins.extend(mins)

        colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']
        x = np.array(list(stabilities_over_time.keys()))
        # plot stabilities over time with peaks and troughs marked
        plt.plot(x, stabilities, label=batch_names[i], color=colors[i])
        if i == len(words_batches) - 1:
            plt.plot(x[mins], np.array(stabilities)[mins], color='#fb9a99', marker='o', linestyle='None', label='mins')
            plt.plot(x[peaks], np.array(stabilities)[peaks], color='#e31a1c', marker='o', linestyle='None',
                     label='peaks')
        else:
            plt.plot(x[mins], np.array(stabilities)[mins], color='#fb9a99', marker='o', linestyle='None')
            plt.plot(x[peaks], np.array(stabilities)[peaks], color='#e31a1c', marker='o', linestyle='None')

    type_analysis = 'diachronic' if mode[:2] == 'd-' else 'synchronic'
    archive = mode[2:] if mode != 's' else None
    # word_proc = bidialg.get_display(arabic_reshaper.reshape(w))
    if archive is not None:
        xlab = 'time points of the stability in {} archive - {} analysis'.format(archive, type_analysis)
    else:
        xlab = 'time points of the stability - {} analysis'.format(type_analysis)
    plt.xlabel(xlab)
    plt.ylabel('stability')
    plt.ylim(-0.5, 1.5)

    cnt1 = Counter(all_maxs)
    cnt2 = Counter(all_mins)
    maxs_to_mark = [k for k, v in cnt1.items() if
                    v == len(words_batches)]  # time points were peaks were found in all batches
    mins_to_mark = [k for k, v in cnt2.items() if
                    v == len(words_batches)]  # time points were troughs were found in all batches

    maxs_to_mark2 = [k for k, v in cnt1.items() if v > 2 and v < 4]
    mins_to_mark2 = [k for k, v in cnt2.items() if v > 2 and v < 4]

    if maxs_to_mark:
        for mx in maxs_to_mark:
            plt.axvline(x=x[mx], color='#fdbf6f', linestyle='--')

    if mins_to_mark:
        for mn in mins_to_mark:
            plt.axvline(x=x[mn], color='#fdbf6f', linestyle='--')

    if maxs_to_mark2:
        for mx in maxs_to_mark2:
            plt.axvline(x=x[mx], color='#ff7f00', linestyle='--')

    if mins_to_mark2:
        for mn in mins_to_mark2:
            plt.axvline(x=x[mn], color='#ff7f00', linestyle='--')

    plt.xticks(rotation=45)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(words_batches), fancybox=True, shadow=True)
    plt.tight_layout()
    # plt.tick_params(axis='x', which='major', pad=1)
    fig = plt.gcf()
    fig.set_size_inches(16, 6)
    mkdir(save_dir)
    plt.savefig(os.path.join(save_dir, fig_name + '.png'))
    plt.savefig(os.path.join(save_dir, fig_name + '.pdf'))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="UBC-NLP-MARBERT", help="model name for archives")

    # neighbor and combined approach (words_file is needed by linear approach as well)
    parser.add_argument("--k", default=100, help="number of nearest neighbors to consider per word - for neighbours and combined approach")
    parser.add_argument("--save_dir", default="results_diachronic_new/", help="directory to save stabilities dictionary")
    parser.add_argument("--threshold", default="0.3,0.4,0.5", help="threshold value(s) for generating contrastive viewpoint summaries")
    parser.add_argument("--cvs_len", default=30, help="length of the contrastive viewpoint summary")  # cvs ==> contrastive viewpoint summary
    args = parser.parse_args()

    model_name = args.model_name
    path_nahar = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/embeddings/An-Nahar/{}/'.format(model_name)
    path_assafir = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/embeddings/As-Safir/{}/'.format(model_name)

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

    # threshold value(s)
    thresh = args.threshold
    if "," in thresh:
        thresholds = thresh.split(",")
        thresholds = [float(t) for t in thresholds]
    else:
        thresholds = [float(thresh)]

    rootdir = "../generate_bert_embeddings/contrastive_summaries"
    categories2words = {}
    for file in os.listdir(rootdir):
        category_name = file.replace(".txt", "")
        categories2words[category_name] = []
        with open(os.path.join(save_dir, file), "r") as f:
            lines = f.readlines()
            for line in lines:
                w = line.strip().replace("\n", "")
                categories2words[category_name].append(w)

    # number of nearest neighbors to include when creating contrastive viewpoint summaries
    knn = int(args.k)
    # length of the contrastive viewpoint summary
    cvs_len = int(args.cvs_len)

    mode = args.mode

    stabilities_over_time = {}
    results_dir = 'evaluate_stability/{}/'.format(mode)

    summaries2save = {}

    for t in thresholds:

        if t not in summaries2save:
            summaries2save[t] = {}

        for year in ['06', '07', '08', '09', '10', '11', '12']:
            if year not in summaries2save[t]:
                summaries2save[t][year] = {}

            for category in categories2words:

                if category not in summaries2save[t][year]:
                    summaries2save[t][year][category] = {}

                for w in categories2words[category]:

                    print('---- word: {} - timepoint: {} ----'.format(w, year))
                    print('threshold: {}'.format(t))

                    # update summary
                    summaries = get_contrastive_viewpoint_summary(w,
                                                                     summary_length=cvs_len,
                                                                     k=k,
                                                                     models=[model_nahar, model_assafir],
                                                                     embeddings=[embeddings_nahar, embeddings_assafir],
                                                                     tokenizers=[tokenizer_nahar, tokenizer_assafir],
                                                                     models_names=[model1_name, model2_name],
                                                                     mat_name=mat_name,
                                                                     year=year,
                                                                     thresh=t)
                    # save updated summary
                    summaries2save[t][year][category][w] = summaries

                    # will save that dictionary every time its updated so that we always keep the latest version
                    # save the dictionary of summaries as a pickle file for later loading
                    save_summary(summary2save=summaries2save, save_dir=results_dir)

