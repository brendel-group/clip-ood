import torch
import numpy as np
import time
from argparse import ArgumentParser
import pyarrow.parquet as pq
import os


def main(args):
    id = args.id
    embedding_name = args.embedding_name  # could be 'ViT-B-16-plus-240__laion400m_e32'
    base_dataset = args.base_dataset  # 'laion200m', 'imagenet-train', 'laion400m'
    eval_dataset = args.eval_dataset
    n_candidates = args.n_candidates
    dir_base = args.dir_base

    base_dir = dir_base+base_dataset+'/'+embedding_name+'/'
    save_dir = dir_base+base_dataset+'/'+embedding_name+'/sims/'+eval_dataset+'/'

    if base_dataset == 'imagenet-train':
        if embedding_name == 'ResNet101':
            base_size = 100000
        elif embedding_name == 'ViT-B-16-plus-240__laion400m_e32':
            base_size = 500000

    # create dir if it doesn't exist
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    #
    if (base_dataset == 'laion400m') and (args.id < 10):
        suffix = '0'+str(id)
    else:
        suffix = str(id)
    #suffix = str(id)

    # Getting query embs and paths
    # embs
    query_embs = torch.from_numpy(np.load(
        dir_base+eval_dataset+'/'+embedding_name+'/images/img_emb_0.npy'))
    device = torch.device("cuda:0")
    query_embs = query_embs.to(device)

    embs = np.load(base_dir + "images/img_emb_" + suffix + ".npy")
    embs = torch.Tensor(embs)

    print(f"embs_shape = {embs.shape}")

    # Paths
    if base_dataset in ['laion400m', 'laion200m']:
        metadata = pq.read_table(base_dir + 'metadata/metadata_' + suffix + '.parquet', use_threads=True)
        paths = np.asarray(metadata['image_path'], dtype=np.int64)
    elif base_dataset == 'imagenet-train':
        paths_start = int(suffix) * base_size

    # Working code (query isn't being batched, only LAION embs are)
    start_time = time.time()

    minibatch_size = int(n_candidates/2)  # always half as much as accumulate
    n_accumulate = minibatch_size * 5
    num_embs = embs.shape[0]
    query_chunk = query_embs.unsqueeze(0)

    # accumulators
    sims_accumulator = torch.zeros((query_embs.shape[0], n_accumulate), dtype=torch.float32)
    sims_accumulator -= 10  # topk issues
    indices_accumulator = torch.zeros((query_embs.shape[0], n_accumulate), dtype=torch.int64)

    # getting mini-batches and their sizes
    n_batchs = (num_embs // minibatch_size)
    leftover = num_embs % minibatch_size
    if leftover != 0:
        n_batchs += 1
    end = 0
    acc_end = 0

    # main computation loop
    for batch_idx in range(n_batchs):
        if (batch_idx == (n_batchs - 1)) and (leftover != 0):
            batch_size = leftover
        else:
            batch_size = minibatch_size

        # start and end for getting the batches of the both sims
        start = end
        end += batch_size
        acc_start = acc_end
        acc_end += batch_size

        # getting the embs chunk
        embs_chunk = embs[start:end, :]
        embs_chunk = embs_chunk.to(device)

        # sims for a batch of sims and whole query
        with torch.no_grad():
            sims_chunk = query_chunk @ embs_chunk.T

        # Accumulating the sims and indices
        sims_accumulator[:, acc_start:acc_end] = sims_chunk.cpu()
        if base_dataset in ['laion400m', 'laion200m']:
            indices_accumulator[:, acc_start:acc_end] = torch.from_numpy(paths[start:end])
        elif base_dataset == 'imagenet-train':
            indices_accumulator[:, acc_start:acc_end] = torch.from_numpy(np.arange(paths_start+start, paths_start+end))

        # If accumulated enough then store best and free up memory and restart
        if (acc_end % n_accumulate) == 0:
            # get best sims and best indices
            _, idx = torch.topk(sims_accumulator, k=n_candidates, axis=-1)

            sims_best = torch.gather(sims_accumulator, -1, idx).cpu().detach()
            indices_best = torch.gather(indices_accumulator, -1, idx).cpu().detach()

            # reinitialize accumulators
            acc_end = n_candidates
            sims_accumulator = torch.zeros((query_embs.shape[0], n_accumulate), dtype=torch.float32)
            sims_accumulator -= 10
            indices_accumulator = torch.zeros((query_embs.shape[0], n_accumulate), dtype=torch.int64)

            sims_accumulator[:, :acc_end] = sims_best
            indices_accumulator[:, :acc_end] = indices_best

            print(f"{end} sims done, total time taken so far is {time.time()-start_time}s")

    # get sims best and indices best
    _, idx = torch.topk(sims_accumulator, k=n_candidates, axis=-1)
    sims_best = torch.gather(sims_accumulator, -1, idx).cpu().detach().numpy()
    indices_best = torch.gather(indices_accumulator, -1, idx).cpu().detach().numpy()

    print(f"{end} sims done, total time taken so far is {time.time() - start_time}s")

    # save everything
    np.save(save_dir + 'sims_' + suffix + '.npy', sims_best)
    np.save(save_dir + 'idcs_' + suffix + '.npy', indices_best)

    query_labels = \
        np.load(dir_base+eval_dataset+'/'+embedding_name+'/labels/labels_0.npy')

    np.save(save_dir + 'query_labels.npy', query_labels)


if __name__ == '__main__':
    # args and directories
    parser = ArgumentParser()
    parser.add_argument('--id', type=int, help='job_id for suffix')
    parser.add_argument('--base_dataset', type=str, help='laion200m, imagenet-train, laion400m')
    parser.add_argument('--eval_dataset', type=str,
                        help='imagenet-val, imagenet-sketch, imagenet-a, imagenet-r, imagenet-v2, objectnet-subsample')
    parser.add_argument('--n_candidates', type=int, help='I generally used 20K for 200M, 10K for 400M, 5K for IN-Train')
    parser.add_argument('--embedding_name', default='ViT-B-16-plus-240__laion400m_e32', help='CLIP embedding model')
    parser.add_argument('--dir_base', type=str)

    args = parser.parse_args()
    main(args)