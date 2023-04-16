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
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve ,auc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


device = "cuda" if  torch.cuda.is_available() else "cpu" 



parser = argparse.ArgumentParser(description='Code for training a lip sync generator via landmark')
parser.add_argument("--batch_size", help="Batch Size", default=8)
parser.add_argument("--tensorboard_dir", default="./tensorboard")
args = parser.parse_args()


def plot_confusion_matrix(gt,pred, class_name):
    cm = confusion_matrix(gt,pred)
    cm_df = pd.DataFrame(cm, index = class_name, columns = class_name)

    # plot fig
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(6,4)
    fig.tight_layout()
    ax.set_title("Confusion Matrix for DINO")
    cm  = sns.heatmap(cm_df, annot=True,cmap="Blues").get_figure()
    plt.show()
    return cm

def roc_plot(pred_probas,gt_label, classes):
    
    # dictionart for False positive rate, True positive rate, and Area Under the curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    #  number of classes
    n_classes = len(classes)
    # generate numnber  from 0 to n classes
    n_labels = np.arange(0, n_classes)
    #  convert labels to one hot encoding 
    labels = label_binarize(gt_label, classes=n_labels)

    # calculate roc curve and area under curve for each class
    for i in range(n_classes):

        # calculate roc curve
        fpr[i] , tpr[i] , _  = roc_curve(labels[:,i],pred_probas[:,i])
        # calcuate area under the curve
        roc_auc[i] = auc(fpr[i], tpr[i])


    # calculate the micro and marco
    fpr["micro"] , tpr["micro"], _ = roc_curve(labels.ravel(), pred_probas.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([ fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):

        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes


    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    # plot fig 
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(9,7)
    # micro line
    ax.plot(fpr["micro"], tpr["micro"], label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]), color='deeppink', linestyle=":", linewidth=4)
    # macro line 
    ax.plot(fpr["macro"], tpr["macro"], label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]), color="navy", linestyle=":", linewidth = 4) 
    # curve line for each classes
    for i in range(n_classes):

        ax.plot( fpr[i], tpr[i], label="ROC curve of class {0} (area = {1:0.2f})".format(  classes[i],roc_auc[i]),)


    ax.plot([0,1],[0,1], "k--", lw=2)
    ax.legend(loc="lower right")
    ax.set_title("Receiving Operating Characteristic (ROC) Curve of DINO")
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    
    plt.show(block=False) 
    roc = ax.get_figure()
    
    return roc
    #fig.set_size_inches(10,10)
    
    #plt.show(block=False)

def compute_knn(backbone, train_loader, test_loader):
    
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

    device =  next(backbone.parameters()).device

    data_loaders = {
       'train'  : train_loader,
       'val'    : test_loader,
    }

    lists = {
        "X_train" : [],
        "y_train" : [],
        "X_val"   : [],
        "y_val"   : []
    }

    for name, data_loader in data_loaders.items():
        for imgs, y in tqdm(data_loader):
            imgs = imgs.to(device)
            lists[f"X_{name}"].append(backbone(imgs).detach().cpu().numpy())
            lists[f"y_{name}"].append(y.detach().cpu().numpy())

    arrays = {k: np.concatenate(l) for k, l in lists.items()}

    estimator = KNeighborsClassifier()
    estimator.fit(arrays["X_train"], arrays["y_train"])
    y_val_pred = estimator.predict(arrays["X_val"])
    
    y_val_pred_proba = estimator.predict_proba(arrays["X_val"])

    acc = accuracy_score(arrays['y_val'], y_val_pred)
    
    cm = plot_confusion_matrix(arrays['y_val'], y_val_pred, [ label_mapping[i] for i in test_loader.dataset.classes])
    
    roc = roc_plot(y_val_pred_proba,arrays["y_val"],  [ label_mapping[i] for i in test_loader.dataset.classes])
    print("Accuracy : " ,acc)

    return acc ,cm ,roc


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
    
    
    writer = SummaryWriter(args.tensorboard_dir)
    
    # load model from whole model
    student_model = torch.load("./checkpoint/dino_best_model.pth", map_location=torch.device(device))

    student_model.eval()

    train, vali ,test = load_data()

    batch_size = args.batch_size

    train_loader = DataLoader(train,batch_size)
    vali_loader  = DataLoader(vali,batch_size)
    test_loader  = DataLoader(test,batch_size)
    
    
    print(test_loader.dataset.classes)
    
    acc, cm , roc = compute_knn(student_model.backbone, train_loader, test_loader)
    
    writer.add_scalar("test/acc",acc,1) 
    writer.add_figure("test/cm",cm,1)
    writer.add_figure("test/roc",roc,1)
    

if __name__ == "__main__":
    main()
