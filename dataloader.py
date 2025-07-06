import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class IDRiD_Dataset(Dataset):
    def __init__(self, image_root, csv_path, mask_root, trainsize, augmentations=False, mask_type='binary'):
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.mask_type = mask_type
        
    
        self.csv = pd.read_csv(csv_path)
        self.csv['Image name'] = self.csv['Image name'].str.replace(".jpg", "", case=False).apply(lambda x: "IDRiD_" + str(int(x.split('_')[-1])).zfill(2))
        available_imgs = set([f.split(".")[0] for f in os.listdir(image_root)])
        self.csv = self.csv[self.csv['Image name'].isin(available_imgs)].reset_index(drop=True)
        self.image_paths = [os.path.join(image_root, f"{img_id}.jpg") for img_id in self.csv['Image name']]
        self.labels = self.csv['Retinopathy grade'].values
        
        
        self.mask_folders = {
            'MA': os.path.join(mask_root, '1. Microaneurysms'),
            'HE': os.path.join(mask_root, '2. Haemorrhages'),
            'EX': os.path.join(mask_root, '3. Hard Exudates'),
            'SE': os.path.join(mask_root, '4. Soft Exudates'),
            'OD': os.path.join(mask_root, '5. Optic Disc')
        }
        
       
        self.base_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4337, 0.2123, 0.0755], std=[0.3116, 0.1665, 0.0866])
        ])
        
       
        self.mask_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()
        ])
        
        # Augmentations
        if self.augmentations:
            print('Using augmentations')
            self.aug_transform = transforms.RandomApply([
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()
            ], p=0.5)
        else:
            print('No augmentations')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
       
        image = Image.open(self.image_paths[index]).convert('RGB')
        label = self.labels[index]
        
        #loading mask
        img_id = os.path.splitext(os.path.basename(self.image_paths[index]))[0]
        masks = []
        
        for lesion, folder in self.mask_folders.items():
            mask_path = os.path.join(folder, f"{img_id}_{lesion}.tif")
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path).convert('L'))
                mask = (mask > 0).astype(np.uint8)
            else:
                mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)  
            masks.append(mask)
        
        masks = np.stack(masks, axis=0)  # (5, H, W)
        
        # Process masks
        if self.mask_type == 'binary':
            final_mask = (np.sum(masks, axis=0) > 0).astype(np.uint8)
            mask_pil = Image.fromarray(final_mask)
        else:
            final_mask = masks  # (5, H, W)
            mask_pil = [Image.fromarray(m) for m in final_mask]

        # Apply base transforms
        image = self.base_transform(image)
        
        if self.mask_type == 'binary':
            mask = self.mask_transform(mask_pil).float()  # (1, H, W)
        else:
            mask = torch.stack([self.mask_transform(m).squeeze() for m in mask_pil]).float()  # (5, H, W)

        return image, torch.tensor(label).long(), mask
