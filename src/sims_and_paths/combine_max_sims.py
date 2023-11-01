import numpy as np
from argparse import ArgumentParser


def main(args):
    # args and directories
    base_dataset_name = args.base_dataset  # imagenet-train or laion200m
    dataset_name = args.eval_dataset
    dir_embeddings = args.dir_embeddings
    embedding_name = args.embedding_name

    base_dir = dir_embeddings + base_dataset_name + '/'+embedding_name + '/sims_max/' + dataset_name + '/'

    # get start and end id
    if base_dataset_name == 'imagenet-train':
        start_id = 0
        end_id = 2
    elif base_dataset_name == 'laion200m':
        start_id = 0
        end_id = 199
    elif base_dataset_name == 'laion400m':
        start_id = 0
        end_id = 447

    sims_accumulator = []
    labels_accumulator = []
    indices_accumulator = []
    for i in range(start_id, end_id+1):

        if (base_dataset_name == 'laion400m') and (i < 10):
            suffix = '0' + str(i)
        else:
            suffix = str(i)
        # load
        sims = np.load(base_dir + 'sims_' + suffix + '.npy', )
        idcs = np.load(base_dir + 'idcs_' + suffix + '.npy')
        labels = np.load(base_dir + 'labels_' + suffix + '.npy')
        sims_accumulator.append(sims)
        indices_accumulator.append(idcs)
        labels_accumulator.append(labels)
        print(f"{i} done")

    # get sims best and indices best
    sims_accumulator = np.concatenate(sims_accumulator)
    labels_accumulator = np.concatenate(labels_accumulator)
    indices_accumulator = np.concatenate(indices_accumulator)

    # save
    np.save(base_dir + 'sims_all.npy', sims_accumulator)
    np.save(base_dir + 'idcs_all.npy', indices_accumulator)
    np.save(base_dir + 'labels_all.npy', labels_accumulator)
    print("Done")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--base_dataset', type=str, help='laion200m, imagenet-train, laion400m')
    parser.add_argument('--eval_dataset', type=str,
                        help='imagenet-val, imagenet-sketch, imagenet-a, imagenet-r, imagenet-v2, objectnet-subsample')
    parser.add_argument('--embedding_name', default='ViT-B-16-plus-240__laion400m_e32', help='CLIP embedding model')
    parser.add_argument('--dir_embeddings', type=str)

    args = parser.parse_args()
    main(args)

