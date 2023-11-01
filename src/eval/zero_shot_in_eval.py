from utils import *
import torchvision.datasets as datasets
import argparse
import open_clip
import numpy as np
import torch
import pickle
import os
import tqdm

# Fill the dictionary of directories where the datasets are in
# dataset_directories = {
#     'imagenet-val': ,
#     'imagenet-r': ,
#     'imagenet-sketch': ,
#     'imagenet-v2': ,
#     'imagenet-a': ,
#     'objectnet-subsample': ,
# }


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


# Calculates accuracy of a clip model given dataloader
def calculate_cnn_accuracy(dataloader, dataset_name, model, device):
    labels = []

    avg_acc = 0
    for img, label in tqdm(dataloader):
        img = img.to(device, non_blocking=True)
        label = label.to(device)

        # compute predictions if imagenet-r or 200 take only 200 logits
        if (dataset_name == 'imagenet-r') or (dataset_name == 'imagenet-200'):
            out = model(img)[:, imagenet_r_mask]
        elif (dataset_name == 'imagenet-a'):
            out = model(img)[:, imagenet_a_mask]
        elif (dataset_name == 'objectnet-subsample'):
            out = model(img)[:, objectnet_subsample_mask]
        else:
            out = model(img)

        out = out.argmax(1)
        acc_batch = ((label == out).sum()).float()
        avg_acc += acc_batch.item()

        # Append predictions and labels
        labels.append(label.cpu().detach())

    # Compute acc
    labels = torch.cat(labels).numpy()
    acc = avg_acc / len(labels)

    return acc


# Calculates accuracy of a clip model given dataloader
def calculate_accuracy(dataloader, model, class_matrix, device):
    labels = []
    predictions = []

    for img, label in tqdm(dataloader):
        img = img.to(device, non_blocking=True)
        img_emb = encode_image(model, img)

        # compute predictions
        pred = torch.argmax(img_emb @ class_matrix.T, dim=-1)

        # Append predictions and labels
        labels.append(label.cpu().detach())
        predictions.append(pred.cpu().detach())

    # Compute acc
    predictions = torch.cat(predictions).numpy()
    labels = torch.cat(labels).numpy()
    acc = np.sum(predictions == labels) / len(labels)

    return acc


def get_dataset(dataset_name, batch_size, num_workers, transform, **kwargs):
    '''Add subset idcs to only test on a subset of the eval dataset
    '''
    datadir = dataset_directories[dataset_name]
    idcs_subset = kwargs.pop('idcs_subset', None)

    # get the full dataset or its subset
    dataset = datasets.ImageFolder(datadir, transform=transform)
    if idcs_subset is not None:
        dataset = torch.utils.data.Subset(dataset, idcs_subset)
        print(f"this is a subset of {dataset_name} with {len(idcs_subset)} datapoints")

    # get the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch_size,
                                             pin_memory=False, num_workers=num_workers)
    return dataloader


# main function
def main(args):
    # model args
    model_name = args['model_name']
    checkpoints_path = args['checkpoints_path']
    checkpoints_epochs = args['checkpoints_epochs']
    pretrained = args['pretrained']
    jit = args['jit']
    cnn = args['cnn']
    
    # data args
    dataset_names = args['dataset_names']
    batch_size = args['batch_size']
    num_workers = args['num_workers']
    idcs_subset_loc = args['idcs_subset_loc']
    
    # subset idcs to evaluate only on a subset
    if idcs_subset_loc is not None:
        idcs_subset = np.load(idcs_subset_loc)
    else:
        idcs_subset = None
    
    # save args
    save_dir = args['save_dir']
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if os.path.exists(save_dir+'accuracy.pickle'):  # resuming zero shot eval
        with open(save_dir + 'accuracy.pickle', 'rb') as handle:
            acc_dict = pickle.load(handle)
    else:
        acc_dict = {}
    
    model, transform = get_model_and_transform(cnn, model_name, pretrained, device, jit)
    
    # For each epoch in a list of epochs (models of different epochs)
    for epoch in checkpoints_epochs:
        if epoch in list(acc_dict.keys()):  # resuming zero shot eval
            continue
        else:
            acc_dict[epoch] = {}

            # Load checkpoint if model weights are given

            if not pretrained:
                model.train() #change mode before load (maybe unnecessary)
                checkpoint_path = checkpoints_path + 'epoch_' + str(epoch) + '.pt'
                checkpoint = torch.load(checkpoint_path)
                sd = checkpoint["state_dict"]
                if next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd, strict = False)
            model.eval()
            print("Model loaded")

            # Compute accuracy on each new dataset
            for dataset_name in dataset_names:
                # Get loader
                dataloader = get_dataset(dataset_name, batch_size, num_workers, transform, idcs_subset=idcs_subset)

                if cnn:
                     acc_dict[epoch][dataset_name] = calculate_cnn_accuracy(dataloader, dataset_name, model, device)

                else:
                    class_matrix = get_class_matrix(model, imagenet_classnames, openai_imagenet_template,
                                                            dataset_name)

                    # Compute and store accuracy
                    acc_dict[epoch][dataset_name] = calculate_accuracy(dataloader, model, class_matrix, device)

                with open(save_dir + 'accuracy.pickle', 'wb') as handle:
                    pickle.dump(acc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                print(f"Epoch: {epoch}, Dataset: {dataset_name}, with acc {acc_dict[epoch][dataset_name]} done")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dir args
    parser.add_argument("-b", "--base_save_dir", type=str, help='saving accuracies')
    parser.add_argument("-n", "--name", type=str,
                        default=None, help='directory')

    # model args
    parser.add_argument("-m", "--model_name", type=str, default='RN50-quickgelu', help='model_name')
    parser.add_argument("-p", "--pretrained", type=boolean_string,
                        default=False, help='using an already pretrained model') # set to True if using a pretrained model(VERY IMP)
    
    parser.add_argument("--cnn", type=boolean_string,
                        default=False, help='using an already pretrained model') # set to True if using a CNN else set to False
    
    parser.add_argument("-c", "--checkpoints_path", type=str, help='model_name')

    # data args
    parser.add_argument("--batch_size", type=int, default=250, help='batch size')
    parser.add_argument("--num_workers", type=int, default=4, help='number of workers for the data loader')
    parser.add_argument("--idcs_subset_loc", type=str, default=None, help='subset idcs to eval on the subset of data')
    parser.add_argument("--dataset", type=str, default=None, help='dataset to be evaluated on')

    ap = parser.parse_args()
    args = vars(ap)

    # data args again

    if args['dataset'] is None:
        args['dataset_names'] = ['imagenet-200', 'imagenet-r', 'imagenet-sketch', 'imagenet-v2', 'imagenet-val']
    else:
        args['dataset_names'] = [args['dataset']]
    
    # This is only for CLIP models
    if args['pretrained']:
        pretrained_data = args['model_name'].split('__')[1]
        if pretrained_data == 'openai':
            args['jit'] = True    # jit = True needed for some openai models
        else:
            args['jit'] = False
        
        args['save_dir'] = args['base_save_dir']+args['model_name']+'/'  # model_name has also the data it's trained on
        if args['name'] is not None:
            args['save_dir'] = args['save_dir'] + args['dataset'] + '/' + args['name'] + '/'

        args['checkpoints_epochs'] = [-1]  # model epochs to be evaluated (Last/Best epoch)
    else:
        args['jit'] = False
        
        args['save_dir'] = args['base_save_dir']+args['model_name']+'_'+args['checkpoints_path'].split('/')[-3]+'/'

        args['checkpoints_epochs'] = [i for i in range(1, len(os.listdir(args['checkpoints_path']))+1)]
        args['checkpoints_epochs'].reverse()   # start from last epoch

    # Create save directory
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])

    # run  main
    print(args)
    main(args)
    print("All done")
