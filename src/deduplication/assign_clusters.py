'''This scripts does kmeans clustering of LAION embeddings and saves all the clusters
'''

import os
import gc
import glob
import pickle
import datetime

import torch
from tqdm import tqdm
from queue import Queue
from argparse import ArgumentParser
import time
from joblib import Parallel, delayed
import pyarrow.parquet as pq
from utils import *


def get_centers_runner(
    file_idx,
    gpu_list_loc,
    cluster_centers,
    img_prefix,
    meta_prefix,
    args
    ):
    """This function takes in current clusters, takes an embedding matrix, assigns clusters to the points in the matrix
       and computes new set of cluster centers.
    """

    # each runner uses a different GPU
    with open(gpu_list_loc, "rb") as fp:  # Pickling
        gpu_working_list = pickle.load(fp)

    gpu_id = gpu_working_list.pop(0)

    with open(gpu_list_loc, "wb") as fp:  # Pickling
        pickle.dump(gpu_working_list, fp)

    start_time = time.time()
    device = torch.device(f'cuda:{gpu_id}')
    print(f"starting process for {file_idx} by gpu {gpu_id}")

    suffix = f'{file_idx:02d}' if args.database == 'laion400m' else f'{file_idx}'

    # move cluster centers to GPU
    #cluster_centers = torch.from_numpy(cluster_centers)
    n_clusters = cluster_centers.shape[0]
    cluster_centers = cluster_centers.to(device, dtype=torch.float32)
    if file_idx == 10:
        print(cluster_centers)

    # load the embedding of a file idx
    datab_img_emb = torch.from_numpy(np.load(img_prefix + suffix + '.npy'))
    n_img_emb = datab_img_emb.shape[0]
    datab_img_emb = datab_img_emb.to(device, dtype=torch.float32)

    # load metadata
    metadata = pq.read_table(meta_prefix + suffix + '.parquet', use_threads=True)
    paths = np.asarray(metadata['image_path'], dtype=np.uint32)

    # both the database batch and query dataset are rather large,
    # so within each database batch we minibatch the target dataset
    # settings for minibatching the target dataset
    minibatch_size = args.batch_size
    n_batchs = (n_img_emb // minibatch_size)
    leftover = n_img_emb % minibatch_size
    if leftover != 0:
        n_batchs += 1
    end = 0

    # matrices to hold the results of minibatching
    cluster_assignments = np.empty(n_img_emb, dtype=int)

    # minibatching loop
    for batch_idx in range(n_batchs):
        if (batch_idx == (n_batchs - 1)) and (leftover != 0):
            batch_size = leftover
        else:
            batch_size = minibatch_size

        start = end
        end += batch_size

        sims = -torch.cdist(cluster_centers, datab_img_emb[start:end, :], p=2.0).detach()

        # assign clusters for this batch
        cluster_assignments[start:end] = sims.argmax(0).detach().cpu().numpy()

    # compute new cluster centers (as a sum)
    clusters_count = []
    for cluster in range(n_clusters):
        locs = (cluster_assignments == cluster)
        clusters_count.append(locs.sum())
        cluster_centers[cluster, :] = torch.sum(datab_img_emb[locs, :], dim=0).detach()

    clusters_count = np.array(clusters_count)
    cluster_centers = cluster_centers.detach().cpu().numpy()
    print(f"{file_idx} done in {time.time()-start_time}s by gpu {gpu_id}")

    with open(gpu_list_loc, "rb") as fp:  # Pickling
        gpu_working_list = pickle.load(fp)

    gpu_working_list.append(gpu_id)

    with open(gpu_list_loc, "wb") as fp:  # Pickling
        pickle.dump(gpu_working_list, fp)

    torch.cuda.empty_cache()
    return cluster_centers, cluster_assignments, paths, clusters_count


def main(args):
    ''' Main function which basically runs k-means on the embedding space of LAION
    '''
    print("Inside main")
    # setup saving/loading settings for the query embeddings and nn ref
    work_dir = args.work_dir

    save_dir = os.path.join(work_dir, args.database, args.model_name, 'clusters',
                            f'{args.n_clusters}_{args.seed}_{args.change_lower_bound}')
    os.makedirs(save_dir, exist_ok=True)

    save_dir_assignments = os.path.join(save_dir, 'assignments')
    os.makedirs(save_dir_assignments, exist_ok=True)

    # To continue k-means from different iteration
    iteration = args.iteration

    # get paths to database embeddings and metadata
    embedding_src_dir = os.path.join(work_dir, 'laion400m', args.model_name)

    # getting image and meta prefix
    img_prefix = os.path.join(embedding_src_dir, 'images', 'img_emb_')
    meta_prefix = os.path.join(embedding_src_dir, 'metadata', 'metadata_')
    
    # getting some arguments
    n_clusters = args.n_clusters
    change_lower_bound = args.change_lower_bound
    seed = args.seed

    # get all the numpy file locs
    npy_files = glob.glob(img_prefix + '*.npy')

    # condition to see if only final clusters need to be computed
    print("Starting procedure")
    # get the initial cluster centers by taking random image embeddings from LAION
    start_time = time.time()

    if iteration is None:
        print("starting init centers computation")
        gpu_id = 0
        device = torch.device(f'cuda:{gpu_id}')
        cluster_centers = get_initial_cluster_centers(n_clusters, npy_files, seed, device)

        np.save(save_dir_assignments + '/cluster_centers_0.npy', cluster_centers)
        print(f"computing init centers took {time.time() - start_time}s")
        torch.cuda.empty_cache()
        iteration = 0
    else:
        cluster_centers = np.load(save_dir_assignments + '/cluster_centers_'+str(iteration)+'.npy')
        if iteration != 0:
            cluster_assignments_old = np.load(save_dir_assignments + '/cluster_assignments_'+str(iteration)+'.npy')
        print(f"loaded init centers from iteration = {iteration}")

    iteration += 1

    start_time = time.time()

    j = 0
    while True:
        cluster_centers = torch.from_numpy(cluster_centers)
        print(f'doing k means iteration = {iteration}')

        # working gpu_list: this is the shared object that saves GPU numbers and loads from it

        if j == 0:
            gpu_working_list = [i for i in range(n_jobs)]
            gpu_list_loc = save_dir_assignments+"/gpu_working_list"
            with open(gpu_list_loc, "wb") as fp:  # Pickling
                pickle.dump(gpu_working_list, fp)
            j = 1

        # here we basically compute cluster assignments (and new rough centers) in parallel for different embeddings
        res = Parallel(n_jobs=n_jobs, prefer='processes')(delayed(get_centers_runner)(
                        i, gpu_list_loc,
                        cluster_centers,
                        img_prefix,
                        meta_prefix,
                        args) for i in tqdm(range(len(npy_files))))

        cluster_centers_list, cluster_assignments_list, paths_list, clusters_count_list = zip(*res)

        # compute new set of centers and concatenate cluster assignments, paths
        cluster_centers = sum(cluster_centers_list)/(sum(clusters_count_list)[:, None])
        cluster_assignments = np.concatenate(cluster_assignments_list, axis=0)
        paths = np.concatenate(paths_list, axis=0)

        # sort paths and cluster assignments

        # sorted_idcs = np.argsort(paths)
        # paths = paths[sorted_idcs]
        # cluster_assignments = cluster_assignments[sorted_idcs]

        # breaking condition (if cluster assignments of many points don't change)
        if iteration != 1:
            change_ratio = 1-((cluster_assignments == cluster_assignments_old).sum()/len(cluster_assignments))
            print(f'after iteration = {iteration}, {change_ratio*100} percent points have changed')
            if change_ratio < change_lower_bound:
                print(f'after iteration = {iteration}, k means has converged, exiting..')
                break

        # save iteration-wise clusters and centers
        np.save(save_dir_assignments + '/cluster_centers_'+str(iteration)+'.npy', cluster_centers)
        np.save(save_dir_assignments + '/cluster_assignments_'+str(iteration)+'.npy', cluster_assignments)
        # np.save(save_dir_assignments + '/cluster_centers_list_'+str(iteration)+'.npy',
        #         np.array(cluster_centers_list))
        np.save(save_dir_assignments + '/cluster_assignments_list_'+str(iteration)+'.npy',
                np.array(cluster_assignments_list))
        np.save(save_dir_assignments + '/paths_list_'+str(iteration)+'.npy', np.array(paths_list))
        np.save(save_dir_assignments + '/paths_'+str(iteration)+'.npy', paths)

        # go to next iteration
        print(f'iteration = {iteration} is done, took {time.time()-start_time}s')
        start_time = time.time()
        iteration += 1
        cluster_assignments_old = cluster_assignments
        # # free up memory (we don't need the query embeddings anymore)
        # del cluster_centers
        # del query_txt_emb
        gc.collect()
        torch.cuda.empty_cache()

    # Save final cluster assignments

    np.save(save_dir_assignments+'/cluster_assignments.npy', cluster_assignments)
    np.save(save_dir_assignments+'/paths.npy', paths)

    args = vars(args)
    with open(save_dir_assignments + '/args.pickle', 'wb') as handle:
        pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # queue with GPU IDs so that parallel threads can use unique GPUs
    n_jobs = torch.cuda.device_count()
    # q = Queue(maxsize=n_jobs)
    # for i in range(n_jobs):
    #     q.put(i)
    print(f"number of jobs is {n_jobs}")

    parser = ArgumentParser()
    parser.add_argument('-d', '--database',
        default='laion400m',
        help='name of the database. one of laion, laion-old (downloaded embeddings), imagenet, imagenet-captions, or yfcc15m')
    parser.add_argument('-m', '--model_name',
        default='ViT-B-32-quickgelu__laion400m_e32',
        help='model used for computing the query embeddings')
    parser.add_argument('--batch_size', type=int, default=10000,
        help='batch size for getting query dataset embeddings')
    parser.add_argument('--num_workers', type=int, default=8,
        help='number of workers for getting dataset embeddings')
    parser.add_argument('--n_clusters', type=int, default=50000,
        help='number of workers for getting dataset embeddings')
    parser.add_argument('--work_dir',
        help='one of perceptual, textual, or combined')
    parser.add_argument('--change_lower_bound', type=float, default=0.0000001,
        help='if less than change_lower_bound ratio points change, then stop k means')
    parser.add_argument('--seed', type=int, default=333,
        help='seed for sorting')

    # If you want to resume k means from a particular iteration use this.
    parser.add_argument('--iteration', type=int, default=None,
        help='None if fresh, 0 if starting from init centers, iteration if anything else')

    args = parser.parse_args()

    main(args)


