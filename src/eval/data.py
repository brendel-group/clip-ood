import logging
import braceexpand
import webdataset as wds
from open_clip import tokenize
import pandas as pd
import torchvision.datasets as datasets


# Directory dicts key-ed with dataset name for ease of use

def preprocess_label(label):
    # the label is stored as a byte string,
    #  so first decode, then convert
    return int(str(label))


def preprocess_txt(text):
    return tokenize([str(text)])[0]


def filter_no_caption(sample):
    return 'txt' in sample


def filter_no_label(sample):
    return 'cls' in sample


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def get_wds_dataloader(input_shards, batch_size, num_workers, train_transform, use_label=False, meta_data=False):
    '''
    input_shards given as data_root+'{00000..41407}.tar'

    return a dataloader that returns an image, and label
    '''

    # shuffle = args['shuffle']   # shuffle not working for wds

    pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    pipeline.extend([
        wds.split_by_worker,
        # at this point, we have an iterator over the shards assigned to each worker
        wds.tarfile_to_samples(handler=log_and_continue),
    ])

    if use_label:
        pipeline.extend([
            wds.select(filter_no_label),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png", label="cls"),
            wds.map_dict(image=train_transform, label=preprocess_label),
            wds.to_tuple("image", "label"),
            wds.batched(batch_size, partial=False),
        ])
    else:
        if meta_data:
            pipeline.extend([
                wds.select(filter_no_caption),
                wds.decode("pilrgb", handler=log_and_continue),
                wds.rename(image="jpg;png", text="txt", metadata="__key__"),
                wds.map_dict(image=train_transform, text=preprocess_txt),
                wds.to_tuple("image", "text", "metadata"),
                wds.batched(batch_size, partial=False),
            ])
        else:
            pipeline.extend([
                wds.select(filter_no_caption),
                wds.decode("pilrgb", handler=log_and_continue),
                wds.rename(image="jpg;png", text="txt"),
                wds.map_dict(image=train_transform, text=preprocess_txt),
                wds.to_tuple("image", "text"),
                wds.batched(batch_size, partial=False),
            ])

    dataset = wds.DataPipeline(*pipeline)

    data_loader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    return data_loader


# Def read parquet file given, integer number

def read_laion_metadata(parquet_no, emb_path):
    if parquet_no < 10:
        parquet_no = '0' + str(parquet_no)
    else:
        parquet_no = str(parquet_no)
    meta_emb = pd.read_parquet(emb_path + 'metadata_' + str(parquet_no) + '.parquet', engine='pyarrow')

    return meta_emb


# For imagenet kinda datasets

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path


