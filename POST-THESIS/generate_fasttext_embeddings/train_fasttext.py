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
    parser.add_argument('--neg', type=int, help='number of negatives sampled')
    parser.add_argument('--dim', type=int, help='embedding vectors dimension')
    parser.add_argument('--year', type=int, help='year to train on')
    parser.add_argument('--month', type=int, default=None, help='month to train on')
    args = parser.parse_args()

    archive = args.archive

    if args.month is not None:
        data_folder = "../original_data/data_splitting/txt_files/{}/monthly/".format(archive)
        logdir = 'trained_embeddings/{}/monthly/dim{}-negative{}/'.format(args.archive, args.dim, args.neg)
    else:
        data_folder = "../original_data/data_splitting/txt_files/{}/yearly/".format(archive)
        logdir = 'trained_embeddings/{}/yearly/dim{}-negative{}/'.format(args.archive, args.dim, args.neg)

    mkdir(logdir)

    print('hello')

    if args.month is not None:
        for file in os.listdir(data_folder):
            if file.endswith('.txt'):
                year = file.split('.')[0].split('-')[0]
                month = file.split('.')[0].split('-')[1]

                if int(year) == args.year and int(month) == args.month:
                    if os.stat(os.path.join(data_folder, file)).st_size == 0:
                        print("file size is 0 -- will not train")
                        break

                    print(f'Processing the data file: {file}')
                    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
                    model = fasttext.train_unsupervised(input=os.path.join(data_folder, file),
                                                        model="skipgram",
                                                        dim=args.dim,
                                                        neg=args.neg,
                                                        thread=mp.cpu_count())

                    model.save_model(os.path.join(logdir, "{}-{}.bin".format(year, month)))
                    break
            else:
                year = file.split('.')[0].split('-')[0]

                if int(year) == args.year:
                    if os.stat(os.path.join(data_folder, file)).st_size == 0:
                        print("file size is 0 -- will not train")
                        break

                    print(f'Processing the data file: {file}')
                    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
                    model = fasttext.train_unsupervised(input=os.path.join(data_folder, file),
                                                        model="skipgram",
                                                        dim=args.dim,
                                                        neg=args.neg,
                                                        thread=mp.cpu_count())

                    model.save_model(os.path.join(logdir, "{}.bin".format(year)))



#  os.stat("file").st_size == 0