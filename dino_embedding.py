from DINO_model import *
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter


device = "cuda" if  torch.cuda.is_available() else "cpu" 



parser = argparse.ArgumentParser(description='Code for training a lip sync generator via landmark')
parser.add_argument("--batch_size", help="Batch Size", default=4)
parser.add_argument("--tensorboard_dir", default="./tensorboard/")
args = parser.parse_args()


def compute_embedding(backbone, data_loader):

    device = next(backbone.parameters()).device

    embs_l = []
    imgs_l = []
    labels_l = []

    with torch.no_grad():
        for img, y in tqdm(data_loader):

            img = img.to(device)
            embs_l.append(backbone(img).detach().cpu())
            imgs_l.append(((img*0.224) + 0.45).cpu())
            labels_l.extend([data_loader.dataset.classes[i] for i in y.tolist()])
            
            del img


        embs = torch.cat(embs_l , dim=0)
        imgs = torch.cat(imgs_l , dim=0)

    return embs, imgs, labels_l




def load_data():

    transformations = transforms.Compose(
                    [
                    transforms.ToTensor(),
                    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
                    transforms.Resize((400,400)),
                    transforms.CenterCrop((224,224)),
                    ]
            )

    train  = ImageFolder('./train_data', transform=transformations)
    vali   = ImageFolder('./vali_data', transform=transformations)
    test   = ImageFolder('./test_data/', transform=transformations)

    return train, vali, test


def main():
    
    label_mapping  = {
        "MEL" : "Melanoma",
        "NV"  :"Melanocytic_nevus",
        "BCC" :"Basal_cell_carcinoma",
        "AK"  :"Actinic_keratosis",
        "BKL" :"Benign_keratosis",
        "DF"  :"Dermatofibroma",
        "VASC":"Vascular_lesion",
        "SCC" :"Squamous_cell_carcinoma",
        "UNK" :"Unknow"
    }

    writer = SummaryWriter(args.tensorboard_dir)
    
   
    # load model from whole model
    student_model = torch.load("./checkpoint/dino_best_model.pth", map_location=torch.device(device))

    student_model.eval()

    train, vali ,test = load_data()

    batch_size = args.batch_size

    train_loader = DataLoader(train,batch_size)
    vali_loader  = DataLoader(vali,batch_size)
    test_loader  = DataLoader(test,batch_size)
    
    embeddings, imgs, labels_l = compute_embedding(student_model.backbone,test_loader)
    
    imgs = transforms.Resize((64,64))(imgs)

    writer.add_embedding(embeddings, metadata=[label_mapping[l] for l in labels_l], label_img = imgs, global_step=1, tag="DINO_embeddings")

    


if __name__ == "__main__":
    main()
