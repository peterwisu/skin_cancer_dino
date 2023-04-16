import pandas as pd
import os
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
import shutil

def map_label(dx):

    labels = {
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

    return labels[dx]

def create_folder(data, name):

    
    for label in data['label'].unique():

        os.makedirs(os.path.join('{}_data'.format(name),label), exist_ok=True)


    for index, row in data.iterrows():

        label = row['label']
        file_name = row['image']
    
        original_path = os.path.join("../prepro_data/","{}.jpg".format(file_name)) # replace "../prepro_data/" with your path of preprocess ISIC data

        shutil.copy(original_path, os.path.join("{}_data".format(name), label, "{}.jpg".format(file_name)))


def main():
    gt_df = pd.read_csv('../ISIC_2019_Training_GroundTruth.csv') # file name ISIC_2019_Training_GroundTruth.csv

    gt_df['label'] = gt_df[gt_df.columns.values[1:]].idxmax(axis=1)

    gt_df['label_name'] = gt_df['label'].map(lambda x: map_label(x))

    train_data, test_data = train_test_split(gt_df, test_size=0.2, random_state=11)
    train_data, vali_data = train_test_split(train_data, test_size=0.2, random_state=11)
    
    print("Creating Folder")
    create_folder(train_data,'train')
    create_folder(vali_data,'vali')
    create_folder(test_data,'test')
    print("Finish")
    

if __name__ == "__main__":


    main()


















