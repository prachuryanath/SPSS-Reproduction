from torch.utils.data import Dataset
from torchvision import transforms
import os
from utils.mytransforms import *
import json


class tn3kDataSet(Dataset):
    def __init__(self, root, expID, mode='train', ratio=10, sign='label', transform=None):
        super(tn3kDataSet, self).__init__()
        self.mode = mode
        self.sign = sign

        # Loading image file paths based on mode and experiment ID
        if mode == 'train':
            if sign =='label':
                if(expID == 1):
                    imgfile = os.path.join(root, 'data/splits/tn3k/322/labeled.txt')
                elif(expID == 2):
                    imgfile = os.path.join(root, 'data/splits/tn3k/644/labeled.txt')
                elif(expID == 3):
                    imgfile = os.path.join(root, 'data/splits/tn3k/1289/labeled.txt')
    
                with open(imgfile,'r') as f:
                    imglist = f.read().splitlines()
                    self.imglist = [os.path.join(root, 'tn3k/trainval-image',img) for img in imglist]
            else:
                if(expID == 1):
                    imgfile = os.path.join(root, 'data/splits/tn3k/322/unlabeled.txt')
                elif(expID == 2):
                    imgfile = os.path.join(root, 'data/splits/tn3k/644/unlabeled.txt')
                elif(expID == 3):
                    imgfile = os.path.join(root, 'data/splits/tn3k/1289/unlabeled.txt')               
                with open(imgfile,'r') as f:
                    imglist = f.read().splitlines()
                    self.imglist = [os.path.join(root, 'tn3k/trainval-image',img) for img in imglist]
        elif mode == 'valid':
            imgfile = os.path.join(root,'data/splits/tn3k/val.txt')
            with open(imgfile,'r') as f:
                    imglist = f.read().splitlines()
                    self.imglist = [os.path.join(root, 'tn3k/trainval-image',img) for img in imglist]
        elif mode == 'test':
            imgfile = os.path.join(root, 'tn3k/test-image')
            self.imglist = [file for file in self.get_all_files(imgfile)]

        # Defining transformations based on mode and label type
        if transform is None:
            if mode == 'train' and sign == 'label':
               transform = transforms.Compose([
                   Resize((320, 320)),
                   RandomHorizontalFlip(),
                   RandomVerticalFlip(),
                   RandomRotation(90),
                   RandomZoom((0.9, 1.1)),
                   RandomCrop((256, 256)),
                   ToTensor()
               ])
            elif mode == 'train' and sign == 'unlabel':
                transform = transforms.Compose([
                    transforms.Resize((320, 320)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(90),
                    transforms.RandomCrop((256, 256)),
                    transforms.ToTensor()
                ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                   Resize((320, 320)),
                   ToTensor()
                ])
        self.transform = transform

    def __getitem__(self, index):
        # Handling unlabeled training images
        if self.mode == 'train' and self.sign == 'unlabel':
            img_path = self.imglist[index]
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                return self.transform(img)

        # Handling test images with ground truth masks
        elif self.mode == 'test' :
            img_path = self.imglist[index]
            gt_path = img_path.replace('test-image', 'test-mask')
            img = Image.open(img_path).convert('RGB')
            gt = Image.open(gt_path).convert('L')
            data = {'image': img, 'label': gt}
    
            if self.transform:
                data = self.transform(data)

            data['name'] = self.imglist[index].split('/')[-1]
 
            return data
        
        # Handling labeled training and validation images with ground truth masks
        else:
            img_path = self.imglist[index]
            gt_path = img_path.replace('trainval-image', 'trainval-mask')
            img = Image.open(img_path).convert('RGB')
            gt = Image.open(gt_path).convert('L')
            data = {'image': img, 'label': gt}
            if self.transform:
                data = self.transform(data)

            data['name'] = self.imglist[index].split('/')[-1]
 
            return data

    def __len__(self):
        #  Return the total number of images
        return len(self.imglist)
    
    def get_all_files(self,directory):
        """
        Recursively retrieves all file paths from the specified directory.
        """
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.abspath(os.path.join(root, file))
                file_paths.append(file_path)
        return file_paths
