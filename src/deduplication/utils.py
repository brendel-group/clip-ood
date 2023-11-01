import random
import copy
import os
import pyarrow.parquet as pq
import glob
import torch
import time
import tarfile
import numpy as np


def get_initial_cluster_centers(n_clusters, npy_file_locs_original, seed, device, threshold=0.9, num_files=100):
    '''This function gets embeddings from the data as initial cluster centers. Ensures there are no duplicate centers.
    '''
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # shuffle files
    npy_file_locs = copy.deepcopy(npy_file_locs_original)
    random.shuffle(npy_file_locs)

    # number of embeddings per file
    num_embs = int(n_clusters / num_files)

    # get embeddings as initial cluster
    query_img_emb = []
    for file_loc in npy_file_locs[:num_files]:
        file = np.load(file_loc)

        # get num_embs random embeddings and append to list
        idcs = np.random.randint(file.shape[0], size=num_embs)
        query_img_emb.append(file[idcs, :])

    # concatenate the list of arrays
    query_img_emb = np.concatenate(query_img_emb, axis=0)
    query_img_emb = torch.from_numpy(query_img_emb).to(device)

    print("got first set of centers, checking for duplicates now")

    # Check if there are duplicates and then replace them
    i = 0
    while True:
        sims = (query_img_emb @ query_img_emb.T).detach()

        # fill upper triangle to be zero
        sims = torch.tril(sims, -1).detach().cpu()

        locs = (sims > threshold).nonzero()
        # gets into this loop if there were duplicate cluster centers
        print(f"{i + 1} round of removing duplicate centers")
        if len(locs) > 0:
            locs_cat = torch.unique(torch.cat([locs[:, 0], locs[:, 1]]))

            file = np.load(npy_file_locs[num_files + i])  # load from unseen tar balls
            idcs = np.random.randint(file.shape[0], size=len(locs_cat))
            file_subset = torch.from_numpy(file[idcs, :]).to(device)

            query_img_emb[locs_cat, :] = file_subset.detach()
            i += 1
        else:
            print(f"No more duplicates found after {i + 1} round, exiting.")
            break

    query_img_emb = query_img_emb.detach().cpu().numpy()
    return query_img_emb


def save_cluster_embeddings(paths_dict, img_prefix, meta_prefix,  dir_clusters_save):
    '''Given cluster assignments and paths as dicts, save clustered embeddings in one file per cluster
    '''

    # create directories
    save_dir_embeddings = os.path.join(dir_clusters_save, 'embeddings')
    os.makedirs(save_dir_embeddings, exist_ok=True)
    save_dir_paths = os.path.join(dir_clusters_save, 'paths')
    os.makedirs(save_dir_paths, exist_ok=True)

    # get npy_files locs
    npy_files = glob.glob(img_prefix + '*.npy')
    cluster_list = list(paths_dict.keys())

    # Creatings dicts for saving
    clusters = {i: [] for i in cluster_list}  # get a dictionary with each an empty list for each cluster
    paths_clusters = {i: [] for i in cluster_list}  # get a dictionary with each an empty list for each cluster

    # main save clusters loop
    # for every file in 414 files
    for file_loc in npy_files:
        start_time = time.time()
        print(f"starting file: {file_loc}")
        # get image embedding and embedding paths
        img_emb = np.load(file_loc)
        file_idx = file_loc.split('_')[-1].split('.')[0]
        metadata = pq.read_table(meta_prefix + file_idx + '.parquet', use_threads=True)
        paths_emb = np.asarray(metadata['image_path'], dtype=np.uint32)

        # for every cluster in num_clusters_iter
        for cl in cluster_list:
            paths_cl = paths_dict[cl]
            intersection_locs = np.intersect1d(paths_cl, paths_emb, return_indices=True)[-1]

            # append all the embeddings/paths for the cluster found in file in corresponding lists
            if len(intersection_locs) > 0:
                clusters[cl].append(img_emb[intersection_locs, :])
                paths_clusters[cl].append(paths_emb[intersection_locs])

        print(f"took {time.time()-start_time}s to form {len(cluster_list)} clusters")

    # One loop around all files is done, so save clusters that have been populated
    for cl in cluster_list:
        clusters[cl] = np.concatenate(clusters[cl], axis=0)
        paths_clusters[cl] = np.concatenate(paths_clusters[cl], axis=0)
        print(f"saving cluster {cl}")
        np.save(save_dir_embeddings + '/' + str(cl) + '.npy', clusters[cl])
        np.save(save_dir_paths + '/' + str(cl) + '.npy', paths_clusters[cl])


def save_sims(paths_dict, dir_clusters_save, device, batch_size):
    ''' compute and save similarities computed within cluster
    '''
    # create directories
    save_dir_embeddings = os.path.join(dir_clusters_save, 'embeddings')
    save_dir_paths = os.path.join(dir_clusters_save, 'paths')
    save_dir_sims = os.path.join(dir_clusters_save, 'sims')
    os.makedirs(save_dir_sims, exist_ok=True)

    # get npy_files locs
    cluster_list = list(paths_dict.keys())

    # main save clusters loop
    clusters = {i: [] for i in cluster_list}  # get a dictionary with each an empty list for each cluster
    paths_clusters = {i: [] for i in cluster_list}  # get a dictionary with each an empty list for each cluster

    # One loop for computing and saving sims
    for cl in cluster_list:
        clusters[cl] = np.load(save_dir_embeddings + '/' + str(cl) + '.npy')
        paths_clusters[cl] = np.load(save_dir_paths + '/' + str(cl) + '.npy')

        start_time = time.time()
        print(f"starting to computing sims for cl: {cl}")
        cluster = torch.from_numpy(clusters[cl]).to(device)
        sims, paths = compute_sims(cluster, paths_clusters[cl], minibatch_size=batch_size)
        # if it was a list
        if isinstance(sims, list):
            for i in range(len(sims)):
                np.save(save_dir_sims + '/' + str(cl) + '_' + str(i) + '.npy', sims[i])
                np.save(save_dir_paths + '/' + str(cl) + '_' + str(i) + '.npy', paths[i])
        else:
            np.save(save_dir_sims + '/' + str(cl) + '.npy', sims)
        print(f"took {time.time()-start_time}s to save sims for  cl: {cl}")


def compute_sims(cluster, paths, minibatch_size=40000):
    '''
    Given a directory and cluster of embeddings in gpu form, saves the similarities.
    '''
    try:
        with torch.no_grad():
            print(cluster.shape)
            sims = cluster @ cluster.T
            sims = torch.tril(sims, diagonal=-1)  # gets only bottom diagonal
            sims = sims.cpu().numpy()
            return sims, paths
    except:
        n_img_emb = cluster.shape[0]
        minibatch_size = minibatch_size
        n_batchs = (n_img_emb // minibatch_size)
        leftover = n_img_emb % minibatch_size
        if leftover != 0:
            n_batchs += 1
        end = 0

        # matrices to hold the results of minibatching
        #sims_np = np.empty((n_img_emb, n_img_emb))
        sims_stack = []
        paths_stack = []
        # minibatching loop
        for batch_idx in range(n_batchs):
            start_time = time.time()
            if (batch_idx == (n_batchs - 1)) and (leftover != 0):
                batch_size = leftover
            else:
                batch_size = minibatch_size

            start = end
            end += batch_size

            paths_stack.append(paths[start:end])
            with torch.no_grad():
                cluster_chunk = cluster[start:end, :]
                sims = cluster_chunk @ cluster.T
                sims = torch.tril(sims, diagonal=start-1)  # get bottom diagonal
                sims_stack.append(sims.cpu().numpy())
            print(f"batch {batch_idx}/{n_batchs} done in {time.time()-start_time}s")
        #sims_stack = np.vstack(sims_stack)  #FIXME: this takes a loooong time (use reshape instead?)
        print(sims_stack)
        return sims_stack, paths_stack


def compute_dedup_paths(sims, paths, similarity):
    sims_max = np.amax(sims, axis=1)
    locs_original = np.where(sims_max <= similarity)[0]
    paths_dedup = paths[locs_original]

    return paths_dedup

# ----------------------------------------------------------------------------------------------------------------------