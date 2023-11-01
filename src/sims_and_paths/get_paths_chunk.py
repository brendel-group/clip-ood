import torch
import numpy as np
import time
import pickle
import os
from argparse import ArgumentParser


def main(args):
    # getting the args ready
    id = args.id
    n_candidates = args.candidates
    best = args.best
    model_name = args.model_name
    base_dataset = args.base_dataset
    eval_dataset = args.eval_dataset
    base_dir_refs = args.base_dir_refs
    dir_embeddings = args.dir_embeddings

    base_dir = dir_embeddings + base_dataset + '/' + model_name + '/sims/' + eval_dataset + '/'
    base_dir_imagenet = dir_embeddings + 'imagenet-train/' + model_name + '/sims/' + eval_dataset

    # getting the query labels, the indices, and the sims

    if os.path.exists(base_dir_imagenet + '/per_query_imagenet_train.npy'):
        per_query = np.load(base_dir_imagenet + '/per_query_imagenet_train.npy')
    elif os.path.exists(base_dir + '/per_query_imagenet_train.npy'):
        per_query = np.load(base_dir + '/per_query_imagenet_train.npy')
    else:
        raise Exception("Per query imagenet train not found")

    dir_refs = os.path.join(base_dir_refs, model_name, 'laion_' + eval_dataset, 'sims_per_query')

    os.makedirs(dir_refs, exist_ok=True)

    # if best is true
    if best:
        nn_sims = np.load(base_dir+'sims_best_'+str(n_candidates)+'.npy')
        nn_paths = np.load(base_dir+'idcs_best_'+str(n_candidates)+'.npy')
    else:
        nn_sims = np.load(base_dir + 'sims_' + str(id) + '.npy')
        nn_paths = np.load(base_dir + 'idcs_' + str(id) + '.npy')

    sims_chunk = nn_sims
    paths_laion_chunk = nn_paths

    # getting the locs where sim range is satisfied

    per_query_chunk = np.expand_dims(per_query, axis=1)  # expand dims for row-wise comparison
    locs_all = (sims_chunk > per_query_chunk).nonzero()

    sims_subset_all = sims_chunk[locs_all]
    paths_subset_all = paths_laion_chunk[locs_all]
    labels_subset_all = locs_all[0]

    # now get unique paths and their idcs. Use them to subset only corresponding sims and labels
    paths_unique, idcs_unique = np.unique(paths_subset_all, return_index=True)
    sims_unique = sims_subset_all[idcs_unique]
    labels_unique = labels_subset_all[idcs_unique]

    # Save all lists
    np.save(dir_refs+'/overall_nn_paths_'+str(id)+'.npy', paths_unique)
    np.save(dir_refs+'/overall_nn_sims_'+str(id)+'.npy', sims_unique)
    np.save(dir_refs+'/overall_nn_labels_'+str(id)+'.npy', labels_unique)
    print('done')


if __name__ == '__main__':
    # main args
    parser = ArgumentParser()
    parser.add_argument('-i', '--id', type=int, default=None, help='job_id for suffix')
    parser.add_argument('-c', '--candidates', type=int, default=10000, help='candidates for best sims')
    parser.add_argument('--best', action='store_true',
                        help='')
    parser.add_argument('--base_dir_refs',
                        help='one of perceptual, textual, or combined')
    parser.add_argument('--dir_embeddings',
                        default='/mnt/lustre/bethge/pmayilvahanan31/embeddings/',
                        help='one of perceptual, textual, or combined')
    parser.add_argument('--eval_dataset', default='imagenet-sketch')
    parser.add_argument('--base_dataset', default='imagenet-sketch')
    parser.add_argument('--model_name', default='ViT-B-16-plus-240__laion400m_e32',
                        help='model used for computing the query and target embeddings')

    args = parser.parse_args()
    main(args)