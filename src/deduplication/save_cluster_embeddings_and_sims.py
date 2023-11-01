'''
This script creates new embeddings from clusters and also computes similarity
'''


import os
import gc
import glob
import pickle

from argparse import ArgumentParser
import time
from utils import *


def main(args):
    print('starting the main function')
    # get all necessary arguments
    work_dir = args.work_dir
    embedding_dir = args.embedding_dir
    iteration = args.iteration
    job_id = args.job_id
    total_jobs = args.total_jobs
    compute_sims_only = args.compute_sims_only
    batch_size = args.batch_size

    dir_assignments = os.path.join(work_dir, 'assignments')
    dir_clusters_save = os.path.join(work_dir, 'clusters', str(iteration))
    print(f"directory save is {dir_clusters_save}")

    os.makedirs(dir_clusters_save, exist_ok=True)

    # get paths to database embeddings and metadata
    embedding_src_dir = os.path.join(embedding_dir, 'laion400m', args.model_name)

    # getting image and meta prefix
    img_prefix = os.path.join(embedding_src_dir, 'images', 'img_emb_')
    meta_prefix = os.path.join(embedding_src_dir, 'metadata', 'metadata_')

    # get the cluster assignments and paths
    cluster_assignments = np.load(dir_assignments + '/cluster_assignments_' + str(iteration) + '.npy')
    total_clusters = len(np.unique(cluster_assignments))
    paths = np.load(dir_assignments + '/paths_' + str(iteration) + '.npy')

    # get path dictionary for specific clusters
    print("starting path computation")

    # For each cluster that we want to save, we first find the paths for it
    start_time = time.time()
    paths_dict = {}
    n_clusters = 0

    # if we are running a job for many clusters then generate dictionary for all of them, else do it for only
    # one cluster
    if total_jobs > 1:
        for cl in range(0+job_id, total_clusters, total_jobs):
            locs = np.where(cluster_assignments == cl)[0]
            paths_dict[cl] = paths[locs]
            n_clusters += 1
    else:
        cl = job_id
        locs = np.where(cluster_assignments == cl)[0]
        paths_dict[cl] = paths[locs]

    print(f"computing paths dict took {time.time()-start_time}s for {n_clusters} clusters")

    device = torch.device('cuda:0')
    if not compute_sims_only:
        save_cluster_embeddings(paths_dict, img_prefix, meta_prefix,  dir_clusters_save)
    print(f"Done with clusters computation and saved them")

    save_sims(paths_dict, dir_clusters_save, device, batch_size)
    print(f"Done with sims computation and saved them")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers for getting dataset embeddings')
    parser.add_argument('--embedding_dir', help='one of perceptual, textual, or combined')

    # arguments that need to be given by the '.sh' script
    parser.add_argument('-m', '--model_name',
                        default='ViT-B-16-quickgelu__laion400m_e32',
                        help='model used for computing the query embeddings')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='5000 for small GPUs and 40000 for large. This is for computing sims')
    parser.add_argument('--work_dir', type=str, default=None,
                        help='directory where assignments are')
    parser.add_argument('--iteration', type=int, default=None,
                        help='This is for which Kmeans iteration we are computing clusters and sims')
    parser.add_argument('--job_id', type=int, default=None,
                        help='current job id (or cluster id)')
    parser.add_argument('--total_jobs', type=int, default=100,
                        help='total parallel jobs running, usually 100, set to 1 if only one job')
    parser.add_argument('--compute_sims_only', action='store_true', default=False,
                        help='if you want to compute only sims')

    args = parser.parse_args()
    main(args)