'''
This script creates new embeddings from clusters and also computes similarity
'''

import os
import torch
from argparse import ArgumentParser
import time
import numpy as np
import glob
from utils import *


def main(args):
    print('starting the main function')
    # get all necessary arguments
    work_dir = args.work_dir
    threshold = args.threshold
    similarity = 1-args.threshold  # threshold is given in the paper, changing it to similarity
    job_id = args.job_id
    total_jobs = args.total_jobs
    n_clusters = args.n_clusters

    # get paths and embeddings directory
    dir_sims = os.path.join(work_dir, 'sims')
    dir_paths = os.path.join(work_dir, 'paths')
    dir_threshold = os.path.join(work_dir, 'paths_'+str(threshold))
    os.makedirs(dir_threshold, exist_ok=True)

    # if only one job then go into this (otherwise if array then go to 'else')
    if total_jobs == 1:
        sims = np.load(dir_sims + '/' + job_id + '.npy')
        paths = np.load(dir_paths + '/' + job_id + '.npy')

        # get dedup paths
        paths_dedup = compute_dedup_paths(sims, paths, similarity)
        np.save(dir_threshold + '/paths_' + job_id + '.npy', paths_dedup)
    else:
        for cl in range(0+int(job_id), n_clusters, total_jobs):
            start_time = time.time()
            try:
                sims = np.load(dir_sims + '/' + str(cl) + '.npy')
                paths = np.load(dir_paths + '/' + str(cl) + '.npy')

                # get dedup paths
                paths_dedup = compute_dedup_paths(sims, paths, similarity)
                np.save(dir_threshold + '/paths_' + str(cl) + '.npy', paths_dedup)

            except:  # goes into this for chunked big clusters
                for chunk in range(len(glob.glob(dir_sims + '/' + str(cl) + '_*.npy'))):  # get all sims with form 'cl_'
                    cl_chunk = str(cl) + '_' + str(chunk)

                    # load sims and paths for the cluster chunk
                    sims = np.load(dir_sims + '/' + cl_chunk + '.npy')
                    paths = np.load(dir_paths + '/' + cl_chunk + '.npy')

                    # get dedup paths
                    paths_dedup = compute_dedup_paths(sims, paths, similarity)
                    np.save(dir_threshold + '/paths_' + cl_chunk + '.npy', paths_dedup)

            print(f"computing paths for took {time.time()-start_time}s for cluster {cl}")
    print(f"Done with all paths computation")


if __name__ == '__main__':
    parser = ArgumentParser()

    # arguments that need to be given by the '.sh' script
    parser.add_argument('--work_dir', type=str, default=None,
                        help='directory where assignments are')
    parser.add_argument('--job_id', type=str, default=None,
                        help='current job id or sims id')
    parser.add_argument('--total_jobs', type=int, default=100,
                        help='total parallel jobs running, usually 100, set to 1 if only one job')
    parser.add_argument('--threshold', type=float, default=0.124,
                        help='dedup threshold')
    parser.add_argument('--n_clusters', type=int, default=50000,
                        help='total number of clusters')

    args = parser.parse_args()
    main(args)