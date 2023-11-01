import os
import numpy as np
import open_clip
from utils import *

# get image_embeddings
base_dir = ''  # directory with image embeddings
dataset_name = 'objectnet-subsample'
img_emb = np.load(base_dir+'images/img_emb_0.npy')
img_emb = torch.from_numpy(img_emb).to(device)

# get class matrix
model_name = 'ViT-B-16-plus-240'
model, _, transforms = open_clip.create_model_and_transforms(model_name, pretrained='laion400m_e32')
model = model.to(device)
class_matrix = get_class_matrix(model, imagenet_classnames, openai_imagenet_template, dataset_name)

# logits and preds
with torch.no_grad():
    logits = img_emb @ class_matrix.T
    preds = torch.argmax(logits, dim=-1)

# make dir and save
logits_dir = base_dir+'logits/'
os.makedirs(logits_dir, exist_ok=True)
np.save(logits_dir+'logits.npy', logits.cpu().numpy())
np.save(logits_dir+'preds.npy', preds.cpu().numpy())
np.save(logits_dir+'class_matrix.npy', class_matrix.cpu().numpy())


