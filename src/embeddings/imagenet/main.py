import time
import numpy as np
import pandas as pd
import argparse
from utils import *
import os
import sys
# from loader import *
# from params import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(args):
    ''' compute the embeddings and predictions for a given dataset
        and return this along with the labels and image paths
    '''
    dataset_name = args.dataset_name
    model_name = args.model_name
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_save_datapoints = args.num_save_datapoints
    save_root = args.save_root
    datadir = args.datadir

    # load the clip model
    pretrained = model_name.split('__')[1]
    model, _, transform = open_clip.create_model_and_transforms(
        model_name.split('__')[0],
        pretrained=pretrained,
        device=device,
        precision='fp16' if pretrained == 'openai' else 'fp32',  # openai models use half precision
        jit=True)
    model.eval()

    # get the dataloader
    dataset = ImageFolderWithPaths(datadir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=batch_size,
        pin_memory=True, num_workers=num_workers)

    paths = []
    img_embs = []
    labels = []
    num_data_points = 0
    save_index = 0
    for img, label, path in tqdm(dataloader):

        # get image embedding
        img = img.to(device, non_blocking=True)
        img_emb = encode_image(model, img)

        # compute predictions

        img_emb = img_emb.cpu().detach()
        label = label.cpu().detach()

        paths += path
        img_embs.append(img_emb)
        labels.append(label)

        num_data_points += len(label)

        if num_data_points % num_save_datapoints == 0:
            print(num_data_points)
            img_embs = torch.cat(img_embs, dim=0)
            labels = torch.cat(labels).numpy()
            paths = np.asarray(paths)

            np.save(save_root+'images/img_emb_'+str(save_index), img_embs)
            np.save(save_root+'labels/labels_'+str(save_index), labels)
            np.save(save_root + 'paths/paths_' + str(save_index), paths)

            save_index += 1

            paths = []
            img_embs = []
            labels = []

    # Last batch if any (only goes loop didn't go in previous chunk) or only one huge chunk
    if (num_data_points % num_save_datapoints != 0) or (save_index == 0):
        print(num_data_points)
        img_embs = torch.cat(img_embs, dim=0)
        labels = torch.cat(labels).numpy()
        paths = np.asarray(paths)

        np.save(save_root + 'images/img_emb_' + str(save_index), img_embs)
        np.save(save_root + 'labels/labels_' + str(save_index), labels)
        np.save(save_root + 'paths/paths_' + str(save_index), paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='yfcc15m', help='imagenet1m/yfcc15m/laion400m')
    parser.add_argument("--datadir", type=str, help='dataset dir')
    parser.add_argument("--model_name", type=str, default='ViT-B-16-plus-240__laion400m_e32', help='model_name')
    parser.add_argument("--save_root", type=str, help='saving predictions')
    parser.add_argument("--batch_size", type=int, default=200, help='batch size')
    parser.add_argument("--num_workers", type=int, default=4, help='number of workers for the data loader')
    parser.add_argument("--num_save_datapoints", type=int, default=500000, help='save after number of samples')
    args = parser.parse_args()

    # Create save directory
    args.save_root = args.save_root+args.dataset_name+'/'+args.model_name+'/'

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root+'/images/')
        os.makedirs(args.save_root+'/labels/')
        os.makedirs(args.save_root+'/paths/')

    # run  main
    main(args)
    print("All done")