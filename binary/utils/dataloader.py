import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import json
import numpy as np

def show_tensor(tensor_list,path=None):
    for i, tensor in enumerate(tensor_list):
        if isinstance(tensor,np.ndarray):
            array=tensor
        else:
            array=np.array(tensor.to('cpu'))
        if path is None:
            np.savetxt('tensor_values'+str(i+1)+'.txt', array, fmt='%0.6f')
        else:
            np.savetxt(path, array, fmt='%0.6f')

# Disable cuDNN optimization
torch.backends.cudnn.enabled = False

class Kits9(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, describe_path, trainsize):
        self.trainsize = trainsize
        
        self.list_sample = [json.loads(x.rstrip()) for x in open(describe_path, 'r')]
        
        # self.images = sorted(self.images)
        # self.gts = sorted(self.gts)
        # self.filter_files()
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),  # Resize the image to the size specified by self.trainsize
            transforms.ToTensor(), # covert to tensor
            transforms.Normalize([0.485, 0.456, 0.406], # Normalize the image data (each pixel in each channel is divided by 255, then subtracted by the mean, and divided by the standard deviation)
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.list_sample)
    
    def __getitem__(self, index):
        this_sample=self.list_sample[index]
        
        image = self.rgb_loader(this_sample['fpath_img'])
        gt = self.binary_loader(this_sample['fpath_segm'])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        show_tensor([gt[0]])
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt


class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(), # convert to tensor
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        x=np.array(gt)
        # show_tensor([np.array(gt)],"before.txt")
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        # show_tensor([gt[0]],"after.txt")
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):

    dataset = PolypDataset(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def __len__(self):
        return self.size

# 接口测试
if __name__ == '__main__':
    describe_path="/zhuzixuan/PraNet/data/medTrainDataset/training.odgt"
    image_root = '/zhuzixuan/PraNet/data/TrainDataset/images/'
    gt_root = '/zhuzixuan/PraNet/data/TrainDataset/masks/'
    # test_dataset = data.DataLoader(Kits9(describe_path, 352))
    test_dataset = data.DataLoader(PolypDataset(image_root, gt_root,352))
    for idx,pack in enumerate(test_dataset):
        img,seg=pack
        print(img.shape)
        print(seg.shape)