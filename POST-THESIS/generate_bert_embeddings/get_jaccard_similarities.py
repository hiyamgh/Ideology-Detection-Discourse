import os
import pickle
from tqdm import tqdm


def jaccard_similarity(str1, str2):
    # Create character trigrams
    def char_ngrams(s, n=3):
        return {s[i:i+n] for i in range(len(s) - n + 1)}

    # Calculate Jaccard similarity between sets of trigrams
    ngrams1, ngrams2 = char_ngrams(str1), char_ngrams(str2)
    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))
    return intersection / union


if __name__ == '__main__':
    archives = ['An-Nahar', 'As-Safir']
    words = []
    for file in os.listdir('entities/bias_quantification/'):
        with open(os.path.join('entities/bias_quantification/', file), encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                w = line.replace('\n', '').strip()
                print(w)
                if len(w.split(" ")) > 1:
                    words.extend(w.split(" "))
                else:
                    words.append(w)

    for file in os.listdir('entities/contrastive_summaries/'):
        with open(os.path.join('entities/contrastive_summaries/', file), encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                w = line.replace('\n', '').strip()
                print(w)
                # words.append(w)
                if len(w.split(" ")) > 1:
                    words.extend(w.split(" "))
                else:
                    words.append(w)

    words_content = []
    for arch in archives:
        path = f'opinionated_articles_DrNabil/1982/txt_files/{arch}/'
        for file in os.listdir(path):
            with open(os.path.join(path, file), encoding='utf-8') as f:
                content = f.read().replace("\n", "").split(" ")
                words_content.extend(content)

    similar = {}
    words_content = list(set(words_content))
    words = list(set(words))
    for w in tqdm(words):
        for wc in words_content:
            try:
                if jaccard_similarity(f'{w}', f'{wc}') >= 0.80:
                    if w in similar:
                        similar[w].add(wc)
                    else:
                        similar[w] = set()
                        similar[w].add(wc)
            except:
                pass

    # Save dictionary to a file
    with open("similarities.pkl", "wb") as file:
        pickle.dump(similar, file)

    # Load dictionary from a pickle file
    with open("similarities.pkl", "rb") as file:
        loaded_dict = pickle.load(file)

    for w in loaded_dict:
        print(w, loaded_dict[w])