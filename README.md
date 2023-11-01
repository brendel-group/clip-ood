# Code
## Compute embeddings
### Compute Imagenet Embedding
Folder: `src/embeddings/imagenet/`

`main.py`: Computes image embeddings, and store labels. No need to compute text embeddings because it's simply just a 1000 x n class matrix.

### Compute LAION Embeddings
We use `https://github.com/rom1504/clip-retrieval` for calculating LAION embeddings.

## DeDuplication of LAION400M 
Folder: `src/deduplication`.
The whole folder is on de-duplicating LAION400M and getting it to 200M datapoints.
The main scripts are as follows:
- `assign_clusters.py`: does k means and assigns clusters to CLIP embeddings
- `save_cluster_embeddings_and_similarities.py`: saves embeddings of the same cluster to a single file.
- `deduplicate.py`: deduplicates and then gives out the deduplicated paths for each cluster, which can be combined for sampling

## Compute similarities between datasets and get paths for pruned datasets
Folder: `src/sims_and_paths`.
Here we compute similarities of eval datasets to LAION in the CLIP embedding space and get paths
- `compute_sims.py` : We first compute similarities for one small laion or imagenet-train embedding chunk to a given dataset and get top k candidates per eval datapoint.
- `combine_sims.py` : We combine all the top k similarities of the chunks to get top k overall candidates in LAION.
- `compute_max_sims.py` : We first compute the max similarity for each datapoint one small laion or imagenet-train embedding chunk to a given dataset.
- `combine_max_sims.py` : Simple concatenating of max sims
- `get_paths_chunk.py`: Using the above candidates that pass the imagenet-train to imagenet-x threshold, we obtain path chunks of sub-sampled dataset.
- `get_paths.ipynb` : We combine the path chunks to get paths for the pruned datasets. 

## Sampling 
Folder: (`src/sampling`).
Run `subsample_dataset.py` with a given paths set (LAION paths) to get the final subsampled LAION dataset.

## Training
For training on all subsampled datasets we use: `https://github.com/mlfoundations/open_clip`. We change total batchsize to 33,600.

## Eval 
Folder: `src/eval`.
Evaluate model on several eval datasets like ImageNet-Sketch/Val/R/V2/A and ObjectNet.