# Does CLIP's Generalization Performance Mainly Stem from High Train-Test Similarity?
This repository provides the codes for all experiments shown in the paper [Does CLIP's Generalization Performance Mainly Stem from High Train-Test Similarity?](https://arxiv.org/abs/2310.09562).

## Setup
- use Python 3.9
- run `pip install -r requirements.txt`

## Compute embeddings
Use code from this folder to compute ImageNet and LAION CLIP embeddings. 
### Compute Imagenet Embedding
Run `src/embeddings/imagenet/main.py` to computes image embeddings, and store labels. No need to compute text embeddings because it's simply just a 1000 x n class matrix.

### Compute LAION Embeddings
We use `https://github.com/rom1504/clip-retrieval` for calculating LAION embeddings.

## DeDuplication of LAION400M 
Run `src/deduplication` scripts to de-duplicate LAION400M and getting it to 200M datapoints.
The main scripts are as follows:
- `assign_clusters.py`: does k means and assigns clusters to CLIP embeddings.
- `save_cluster_embeddings_and_similarities.py`: saves embeddings of the same cluster to a single file.
- `deduplicate.py`: deduplicates and then gives out the deduplicated paths for each cluster, which can be combined for sampling.

## Compute similarities between datasets and get paths for pruned datasets
Run `src/sims_and_paths` scripts to compute similarities of eval datasets to LAION in the CLIP embedding space and get paths.
- `compute_sims.py` : compute similarities for one small laion or imagenet-train embedding chunk to a given dataset and get top k candidates per eval datapoint.
- `combine_sims.py` : combine all the top k similarities of the chunks to get top k overall candidates in LAION.
- `compute_max_sims.py` : compute the max similarity for each datapoint one small laion or imagenet-train embedding chunk to a given dataset.
- `combine_max_sims.py` : simple concatenating of max sims.
- `get_paths_chunk.py`: using the above candidates that pass the imagenet-train to imagenet-x threshold, we obtain path chunks of sub-sampled dataset.
- `get_paths.ipynb` : combine the path chunks to get paths for the pruned datasets. 

## Sampling 
Run `src/sampling/subsample_dataset.py` with a given paths set (LAION paths in .npy format) to get the final subsampled LAION dataset.

## Training
For training on all subsampled datasets we use: `https://github.com/mlfoundations/open_clip`. We change total batchsize to 33,600.

## Eval 
Run `src/eval` scripts to evaluate model on several eval datasets like ImageNet-Sketch/Val/R/V2/A and ObjectNet.

## Citation
If you find the insights from the paper or our code base useful, please cite:
```
@inproceedings{mayilvahanan2024does,
      title={Does CLIP's Generalization Performance Mainly Stem from High Train-Test Similarity?},
      author={Prasanna Mayilvahanan and Thadd{\"a}us Wiedemer and Evgenia Rusak and Matthias Bethge and Wieland Brendel},
      booktitle={The Twelfth International Conference on Learning Representations},
      year={2024},
      url={https://openreview.net/forum?id=tnBaiidobu}
}
```
