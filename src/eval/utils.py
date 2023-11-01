from data import *
import open_clip
from imagenet_data import *
import torch

def encode_image(model, image):
    ''' encode an image using CLIP
    '''

    with torch.no_grad():
        embedding = model.encode_image(image)
    embedding /= embedding.norm(dim=-1, keepdim=True)

    return embedding


def encode_text(model, text):
    ''' encode a list of text prompts using CLIP
    '''

    with torch.no_grad():
        embedding = model.encode_text(open_clip.tokenize(text, truncate=True).to(device))
    embedding /= embedding.norm(dim=-1, keepdim=True)

    return embedding


def get_class_matrix(model, imagenet_classnames, templates, eval_dataset):
    ''' gets a matrix that contains a class embedding
        averaged over several different prompts
        this is the OpenAI setting
    '''
    classnames = []
    if eval_dataset == 'objectnet-subsample':
        classnames = objectnet_classnames_sorted
    else:
        for i in range(len(imagenet_classnames)):
            if eval_dataset == 'imagenet-r':
                if imagenet_r_mask[i]:
                    classnames.append(imagenet_classnames[i])
            elif eval_dataset == 'imagenet-a':
                if imagenet_a_mask[i]:
                    classnames.append(imagenet_classnames[i])
            else:
                classnames.append(imagenet_classnames[i])

    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template(classname) for template in templates]  # format with class
            texts = open_clip.tokenize(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = torch.nn.functional.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.T  # (num_cls, embed_dim)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


###--------------------------------------------Return model and transforms------------------------------------###

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, \
    ResNet152_Weights


# Ugly function to get models
def get_model_and_transform(cnn, model_name, pretrained, device, jit):
    if cnn:
        cnn_name = model_name.split('__')[0]
        if cnn_name == 'resnet18':
            transform = ResNet18_Weights.DEFAULT.transforms()
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)

        elif cnn_name == 'resnet34':
            transform = ResNet34_Weights.DEFAULT.transforms()
            model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).to(device)

        elif cnn_name == 'resnet50':
            transform = ResNet50_Weights.DEFAULT.transforms()
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)

        elif cnn_name == 'resnet101':
            transform = ResNet101_Weights.DEFAULT.transforms()
            model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1).to(device)

        elif cnn_name == 'resnet152':
            transform = ResNet152_Weights.DEFAULT.transforms()
            model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1).to(device)


    else:
        if not pretrained:
            pretrained_data = ''
        else:
            pretrained_data = model_name.split('__')[1]

        model, _, transform = open_clip.create_model_and_transforms(
            model_name.split('__')[0],
            pretrained=pretrained_data,
            device=device,
            precision='fp16' if pretrained_data == 'openai' else 'fp32',  # openai models use half precision
            jit=jit)

    return model, transform


