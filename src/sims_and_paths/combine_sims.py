import torch
import numpy as np
import time
from argparse import ArgumentParser


def main(args):
    # important args
    embedding_name = args.embedding_name  # could be 'ViT-B-16-plus-240__laion400m_e32'
    base_dataset = args.base_dataset  # 'laion200m', 'imagenet-train', 'laion400m'
    eval_dataset = args.eval_dataset
    dir_base = args.dir_base
    best = args.worst  # getting the best or worst set
    base_dir = dir_base + base_dataset + '/' + embedding_name + '/sims/' + eval_dataset + '/'
    n_candidates_final = args.n_candidates

    embs_temp = np.load(base_dir + 'sims_0.npy')
    n_candidates = embs_temp.shape[-1]  # gets the candidates/shape of a sims embedding
    shape_eval_dataset = embs_temp.shape[0]

    # populate the embedding numbers
    if base_dataset == 'laion400m':
        start_emb = 0
        end_emb = 447
    elif base_dataset == 'laion200m':
        start_emb = 0
        end_emb = 199
    elif base_dataset == 'imagenet-train':
        start_emb = 0
        end_emb = 2

    emb_nos = [i for i in range(start_emb, end_emb + 1)]

    # Working code (query isn't being batched, only LAION logits are)
    start_time = time.time()

    # accumulators
    n_accumulate = n_candidates * 6
    sims_accumulator = torch.empty((shape_eval_dataset, n_accumulate), dtype=torch.float32)
    indices_accumulator = torch.empty((shape_eval_dataset, n_accumulate), dtype=torch.int64)

    if best:
        sims_accumulator -= 10
    else:
        sims_accumulator += 10

    acc_end = 0

    # main computation loop
    for emb_no in emb_nos:
        suffix = str(emb_no)

        acc_start = acc_end
        acc_end += n_candidates

        # sims and idcs chunks
        sims_chunk = np.load(base_dir + 'sims_' + suffix + '.npy')
        ids_chunk = np.load(base_dir + 'idcs_' + suffix + '.npy')

        # Accumulating the sims/ids
        sims_accumulator[:, acc_start:acc_end] = torch.from_numpy(sims_chunk)
        indices_accumulator[:, acc_start:acc_end] = torch.from_numpy(ids_chunk)

        # If accumulated enough then store best and free up memory and restart
        if acc_end % n_accumulate == 0:
            # get sims best and indices best
            if best:
                _, idx = torch.topk(sims_accumulator, k=n_candidates_final, axis=-1)
            else:
                _, idx = torch.topk(-sims_accumulator, k=n_candidates_final, axis=-1)

            sims_best = torch.gather(sims_accumulator, -1, idx).cpu().detach()
            indices_best = torch.gather(indices_accumulator, -1, idx).cpu().detach()

            # reinitialize accumulators
            acc_end = n_candidates_final
            sims_accumulator = torch.empty((shape_eval_dataset, n_accumulate), dtype=torch.float32)
            indices_accumulator = torch.empty((shape_eval_dataset, n_accumulate), dtype=torch.int64)
            if best:
                sims_accumulator -= 10
            else:
                sims_accumulator += 10

            sims_accumulator[:, :acc_end] = sims_best
            indices_accumulator[:, :acc_end] = indices_best

    # last loop
    # get sims best and indices best
    if best:
        _, idx = torch.topk(sims_accumulator, k=n_candidates_final, axis=-1)
    else:
        _, idx = torch.topk(-sims_accumulator, k=n_candidates_final, axis=-1)
    sims_best = torch.gather(sims_accumulator, -1, idx).cpu().detach().numpy()
    indices_best = torch.gather(indices_accumulator, -1, idx).cpu().detach().numpy()

    print(f"{len(emb_nos)} embs done, total time taken so far is {time.time() - start_time}s for {eval_dataset}")
    # save the best sims with combined final list of candidates
    if best:
        np.save(base_dir + 'sims_best_' + str(n_candidates_final) + '.npy', sims_best)
        np.save(base_dir + 'idcs_best_' + str(n_candidates_final) + '.npy', indices_best)
        if base_dataset == 'imagenet-train':
            np.save(base_dir + 'per_query_imagenet_train.npy', sims_best[:, 0])
        else:
            np.save(base_dir + 'per_query_best_sims.npy', sims_best[:, 0])
    else:
        np.save(base_dir + 'sims_worst_' + str(n_candidates_final) + '.npy', sims_best)
        np.save(base_dir + 'idcs_worst_' + str(n_candidates_final) + '.npy', indices_best)
        np.save(base_dir + 'per_query_worst_sims.npy', sims_best[:, 0])


if __name__ == '__main__':
    # args and directories
    parser = ArgumentParser()
    parser.add_argument('--id', type=int, help='job_id for suffix')
    parser.add_argument('--base_dataset', type=str, help='laion200m, imagenet-train, laion400m')
    parser.add_argument('--eval_dataset', type=str,
                        help='imagenet-val, imagenet-sketch, imagenet-a, imagenet-r, imagenet-v2, objectnet-subsample')
    parser.add_argument('--n_candidates', type=int, help='I generally used 20K for 200M, 10K for 400M, 5K for IN-Train')
    parser.add_argument('--embedding_name', help='CLIP embedding model')
    parser.add_argument('--dir_base', type=str)
    parser.add_argument('--worst', action='store_false', help='if you want to get the worst sims')

    args = parser.parse_args()
    main(args)
