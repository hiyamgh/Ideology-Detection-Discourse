import fasttext
import argparse
import os, logging
import multiprocessing as mp


def mkdir(folder):
    ''' creates a directory if it doesn't already exist '''
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=str, default='assafir', help="name of the archive to transform")
    parser.add_argument('--neg', type=int, default=15, help='number of negatives sampled')
    parser.add_argument('--dim', type=int, default=300, help='embedding vectors dimension')
    parser.add_argument('--year', type=int, help='year to train on')
    parser.add_argument("--train_file", type=str, help=".txt file to train over")
    args = parser.parse_args()

    archive = args.archive
    year = args.year
    file = args.train_file
    neg = args.neg
    dim = 300
    minCount = 300
    wordNgrams = 4
    ws = 5
    lr = 0.001

    logdir = f"trained_models/{archive}/{year}-neg{neg}-dim{dim}-minCount{minCount}-wordNgrams{wordNgrams}-ws{ws}-lr{lr}/"
    mkdir(logdir)

    print(f'Processing the data file: {file}')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = fasttext.train_unsupervised(input=file,
                                        model="skipgram",
                                        dim=300,
                                        minCount=300,
                                        wordNgrams=4,
                                        neg=args.neg,
                                        ws=5,
                                        lr=0.001,
                                        thread=mp.cpu_count())

    model.save_model(os.path.join(logdir, "{}.bin".format(year)))