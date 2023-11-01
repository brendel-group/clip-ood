import torch
import numpy as np
import time
from argparse import ArgumentParser
import pyarrow.parquet as pq
import os


def main(args):
    # args and directories
    base_dataset_name = args.base_dataset  # imagenet-train or laion200m
    dataset_name = args.eval_dataset
    dir_embeddings = args.dir_embeddings
    embedding_name = args.embedding_name
    id = args.id

    base_dir = dir_embeddings + base_dataset_name + '/'+embedding_name+'/'
    save_dir = dir_embeddings + base_dataset_name + '/'+embedding_name+'/sims_max/' + dataset_name + '/'

    # create dir if it doesn't exist
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Getting embs and paths
    query_embs = torch.from_numpy(np.load(
        dir_embeddings + dataset_name + '/ViT-B-16-plus-240__laion400m_e32/images/img_emb_0.npy'))
    query_labels = np.load(
        dir_embeddings + dataset_name + '/ViT-B-16-plus-240__laion400m_e32/labels/labels_0.npy')

    if (base_dataset_name == 'laion400m') and (args.id < 10):
        suffix = '0' + str(id)
    else:
        suffix = str(id)

    embs = np.load(base_dir + "images/img_emb_" + suffix + ".npy")
    minibatch_size = 10000  # batch size for sims computation

    embs = torch.Tensor(embs)

    # Paths
    if base_dataset_name in ['laion200m', 'laion400m']:
        metadata = pq.read_table(base_dir + 'metadata/metadata_' + suffix + '.parquet', use_threads=True)
        paths = np.asarray(metadata['image_path'], dtype=np.int64)
    else:
        paths = np.load(base_dir + 'paths/paths_' + suffix + '.npy')

    # Getting the two basic objects to gpu device
    device = torch.device("cuda:0")
    query_embs = query_embs.to(device)

    print(f"embs_shape = {embs.shape}")

    num_embs = embs.shape[0]

    # accumulators
    sims_max_accumulator = []
    labels_accumulator = []
    indices_accumulator = []

    # getting minibatches and their sizes
    n_batchs = (num_embs // minibatch_size)
    leftover = num_embs % minibatch_size
    if leftover != 0:
        n_batchs += 1
    end = 0
    start_time = time.time()
    # main computation loop
    for batch_idx in range(n_batchs):
        if (batch_idx == (n_batchs - 1)) and (leftover != 0):
            batch_size = leftover
        else:
            batch_size = minibatch_size

        # start and end for getting the batches of the both sims
        start = end
        end += batch_size

        # getting the logit chunk
        embs_chunk = embs[start:end, :]
        embs_chunk = embs_chunk.to(device)

        # sims for a batch of sims and whole query
        with torch.no_grad():
            sims_chunk = query_embs @ embs_chunk.T
            sims_max, labels = torch.max(sims_chunk, dim=0)

        sims_max_accumulator.append(sims_max.cpu().numpy())
        labels_accumulator.append(labels.cpu().numpy())
        indices_accumulator.append(paths[start:end])

        if (end % minibatch_size) == 0:
            print(f"{end} embs done in {time.time() - start_time}s")
            start_time = time.time()

    # get sims best and indices best
    sims_max_accumulator = np.concatenate(sims_max_accumulator)
    labels_accumulator = np.concatenate(labels_accumulator)
    indices_accumulator = np.concatenate(indices_accumulator)

    # save
    np.save(save_dir + 'sims_' + suffix + '.npy', sims_max_accumulator)
    np.save(save_dir + 'idcs_' + suffix + '.npy', indices_accumulator)
    np.save(save_dir + 'labels_' + suffix + '.npy', labels_accumulator)
    np.save(save_dir + 'query_labels.npy', query_labels)

    print("Done")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--id', type=int, help='job_id for suffix')
    parser.add_argument('--base_dataset', type=str, help='laion200m, imagenet-train, laion400m')
    parser.add_argument('--eval_dataset', type=str,
                        help='imagenet-val, imagenet-sketch, imagenet-a, imagenet-r, imagenet-v2, objectnet-subsample')
    parser.add_argument('--embedding_name', default='ViT-B-16-plus-240__laion400m_e32', help='CLIP embedding model')
    parser.add_argument('--dir_embeddings', type=str)

    args = parser.parse_args()
    main(args)

