import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from skmultilearn.model_selection import IterativeStratification
from collections import Counter

class COCODataset(Dataset):
    def __init__(self, root_dir, list_file, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            annotation_file (string): Path to the json file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = (root_dir)
        self.transform = transform
        self.image_paths = []
        print(root_dir)
        with open(list_file, 'r') as file:
            listed_images = file.read().splitlines()
        listed_images_set = set(listed_images)

        for path, subdirs, files in os.walk(self.root_dir):
            for name in files:
                if name in listed_images_set:
                #if name.lower().endswith((".jpg", ".jpeg")):#".jpg", ".jpeg", ".png"
                    self.image_paths.append(os.path.join(path, name))
        print(f"\nTotal number of images found: {len(self.image_paths)}")
       
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name).convert('RGB')

        # If the transform variable is not None
        # then it applies the transformations.
        if self.transform:
            y1, y2 = self.transform(image)
        else:
            to_tensor = transforms.ToTensor()
            y1, y2 = to_tensor(image), to_tensor(image)

        return (y1, y2), idx


class ChestXrayDataset(Dataset):
    def __init__(self, img_dir, csv_file, test_list_file,
                  split_type='train', transform=None, test_size=0.2,
                    random_state=42, select_percent=100):
        """
        img_dir: Directory with all the images.
        csv_file: CSV file containing image paths and labels
        test_list_file: File containing test set image names
        transform: Transformations applied to images
        """
        self.img_dir = img_dir
        self.transform = transform
        self.split_type = split_type
        self.random_state = random_state

        # Read CSV file
        df_annotation_all = pd.read_csv(csv_file)
        # Read the test set list
        with open(test_list_file, 'r') as file:
            test_list = [line.strip() for line in file.readlines()]
        # Filter out only images in the test set
        self.df_annotation_test = df_annotation_all[df_annotation_all['Image Index'].isin(test_list)]
        print(f"\nTotal number of images found: {len(self.df_annotation_test)}")

        # Define category list
        self.classes = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 
                   'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 
                   'Mass', 'Hernia', 'No Finding']
        self.label_to_index = {label: index for index, label in enumerate(self.classes)}

        # Perform multi-hot encoding and arrange it in a specific order
        labels = self.df_annotation_test['Finding Labels'].str.get_dummies(sep='|').reindex(columns=self.classes, fill_value=0)

        #sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[test_size,1.0 - test_size])#No shuaffle
        train_indexes, test_indexes = next(stratifier.split(np.zeros(labels.shape[0]), labels))

        # select a certain proportion of samples
        print(select_percent)
        if split_type == 'train' and select_percent < 100:
            train_stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[select_percent / 100.0, 1.0 - select_percent / 100.0])
            _, sub_train_indexes = next(train_stratifier.split(np.zeros(len(train_indexes)), labels.iloc[train_indexes]))
            self.df = self.df_annotation_test.iloc[train_indexes[sub_train_indexes]]
        elif split_type == 'train'and select_percent == 100:
            self.df = self.df_annotation_test.iloc[train_indexes]
        elif split_type == 'test':
            self.df = self.df_annotation_test.iloc[test_indexes]
        else:
            raise Exception('train_percent in {1,10,100},test_percent awalys {100}')
        
        print(f"\nTotal number of images In {self.split_type} : {len(self.df)}")
                #labels = self.df_annotation_test['Finding Labels']  # 假设标签在'Finding Labels'列
        '''
        for train_index, test_index in sss.split(self.df_annotation_test, labels):
            if split_type == 'train':
                self.df = self.df.iloc[train_index]
            elif split_type == 'test':
                self.df = self.df.iloc[test_index]
        '''

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['Image Index']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        label_str = self.df.iloc[idx]['Finding Labels']
        label_indices = [self.label_to_index[label] for label in label_str.split('|') if label in self.label_to_index]
        label_tensor = torch.zeros(len(self.label_to_index))
        label_tensor[label_indices] = 1

        if self.transform:
            image = self.transform(image)
        label_tensor = label_tensor.clone().detach()
        return image, label_tensor

if __name__ == '__main__':
    img_dir = '/data/public_data/NIH/images-224/images-224'
    csv_file = '/data/public_data/NIH/Data_Entry_2017.csv'
    test_list_file = '/data/public_data/NIH/test_list_NIH.txt'
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # Create training and test set
    train_dataset = ChestXrayDataset(img_dir, csv_file, test_list_file, 
                                     split_type='train', transform=transform, select_percent=1)
    test_dataset = ChestXrayDataset(img_dir, csv_file, test_list_file, 
                                    split_type='test', transform=transform, select_percent=100)
    print(train_dataset.classes)

    def get_label_distribution(dataset):
        label_counter = Counter()
        for _, label_tensor in dataset:
            # label_tensor is a multi-hot encoded tensor
            labels = label_tensor.numpy()
            for i, label_present in enumerate(labels):
                if label_present:
                    label_name = dataset.classes[i]
                    label_counter[label_name] += 1
        return label_counter

    # Get and print the label distribution 
    train_label_distribution = get_label_distribution(train_dataset)
    test_label_distribution = get_label_distribution(test_dataset)

    print("Train Dataset Label Distribution:")
    for label, count in train_label_distribution.items():
        print(f"{label}: {count}")

    print("\nTest Dataset Label Distribution:")
    for label, count in test_label_distribution.items():
        print(f"{label}: {count}")


'''
Label	Train Count	Test Count	Test Proportion
Hernia	60	26	30.23%
Infiltration	4262	1826	29.99%
No Finding	6900	3022	30.46%
Emphysema	765	328	30.01%
Pneumothorax	1863	798	29.99%
Pleural_Thickening	841	304	26.55%
Effusion	3254	1394	29.99%
Pneumonia	334	143	29.98%
Mass	1198	514	30.02%
Cardiomegaly	745	319	29.98%
Edema	647	277	29.98%
Atelectasis	2279	977	30.01%
Consolidation	1270	544	29.99%
Fibrosis	304	130	29.95%
Nodule	1130	484	29.99%

'''