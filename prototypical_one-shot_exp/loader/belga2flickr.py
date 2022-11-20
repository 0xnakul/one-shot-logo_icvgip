import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.misc as m
# from augmentations import *
# from models import get_model
import random

import random
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path

from PIL import Image
random.seed(1)

cpu = torch.device('cpu')

# Transformations borrowed from BarlowTwins repo



class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self, img_size=64):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size,
                                         interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(img_size,
                                         interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


supcon_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(64,64), scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ])


class SupConTwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self):
        self.transform = supcon_train_transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class OpenLogoTrain(torch.utils.data.Dataset):
    def __init__(self, img_lbl_pairs, txt_features_pth=None,
                 no_txt_features_pth=None, img_size=224):
        super(OpenLogoTrain, self).__init__()
        self.img_paths, self.labels = tuple(zip(*img_lbl_pairs))
        self.transform = Transform(img_size)

        self.use_text_features = txt_features_pth is not None

        if txt_features_pth is not None:
            self.txt_features = torch.load(txt_features_pth,
                                           map_location=cpu)
            self.no_txt_feature = torch.load(no_txt_features_pth,
                                             map_location=cpu)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        f = self.img_paths[idx]

        if self.use_text_features:
            fname = Path(f).stem
            feat_vecs = self.txt_features.get(fname)

            if feat_vecs is not None:
                txt_feature = torch.tensor(feat_vecs)
            else:
                txt_feature = self.no_txt_feature

        f = Image.open(f)
        f1, f2 = self.transform(f)

        if self.use_text_features:
            return f1, f2, self.labels[idx], txt_feature.float()

        return f1, f2, self.labels[idx]


class OpenLogoVal(Dataset):

    def __init__(self, val_pairs_path, transform=None, txt_features_pth=None,
                 no_txt_features_pth=None):

        self.val_pairs = np.load(val_pairs_path)
        self.transform = transform

        self.use_text_features = txt_features_pth is not None

        if txt_features_pth is not None:
            self.txt_features = torch.load(txt_features_pth,
                                           map_location=cpu)
            self.no_txt_feature = torch.load(no_txt_features_pth,
                                             map_location=cpu)

    def __getitem__(self, index):

        pth_img1 = self.val_pairs[index][0]
        pth_img2 = self.val_pairs[index][1]

        img0 = Image.open(pth_img1)
        img1 = Image.open(pth_img2)

        if self.use_text_features:
            fname = Path(pth_img1).stem
            feat_vecs = self.txt_features.get(fname)

            if feat_vecs is not None:
                txt_features1 = torch.tensor(feat_vecs)
            else:
                txt_features1 = self.no_txt_feature

            # for img2
            fname = Path(pth_img2).stem
            feat_vecs = self.txt_features.get(fname)

            if feat_vecs is not None:
                txt_features2 = torch.tensor(feat_vecs)
            else:
                txt_features2 = self.no_txt_feature

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        if self.use_text_features:
            return {'image1': img0, 'image2': img1,
                    'txt_vecs1': txt_features1.float(),
                    'txt_vecs2': txt_features2.float(),
                    'label': torch.tensor(float(self.val_pairs[index][2]))}

        return {'image1': img0, 'image2': img1,
                'label': torch.tensor(float(self.val_pairs[index][2]))}

    def __len__(self):
        # return 10 # Debug only
        return len(self.val_pairs)


class belga2flickrLoader(Dataset):

  def __init__(self, root, exp, split='train', is_transform=False, img_size=None, augmentations=None, prototype_sampling_rate = 0.005, use_txt=False, is_supcon=False):
    super().__init__()

    self.use_txt = use_txt
    if split == 'train':
        self.proto_rate = prototype_sampling_rate
    else:
        self.proto_rate = 0.0
    self.inputs = []
    self.targets = []
    self.class_names = []

    if split == 'train':
        self.split = 'belga'
        self.n_classes = 37 #
        self.tr_class = torch.LongTensor([12, 16, 28, 30]) # belga classes
        self.te_class = torch.LongTensor([0,1,2,3,4,5,6,7,8,9,10,11,  13,14,15,  17,18,19,20,21,22,23,24,25,26,27,  29,  31,32,33,34,35,36]) # belga classes
    elif split == 'val':
        self.split = 'toplogo10'
        self.n_classes = 11
        self.tr_class = torch.LongTensor(range(0,11))
        self.te_class = torch.LongTensor(range(0,11))
    elif split == 'test':
        self.split = 'flickr32'
        self.n_classes = 32
        self.tr_class = torch.LongTensor([7, 13, 25, 28]) # belga classes
        self.te_class = torch.LongTensor([0,1,2,3,4,5,6,  8,9,10,11,12,  14,15,16,17,18,19,20,21,22,23,24,  26,27,  29,30,31]) # belga classes
    
    if use_txt:
        self.jpg_txt_features = torch.load(f'./db/text_vecs/{self.split}/jpg_lstm_emb.pt', map_location=cpu)
        self.templ_txt_features = torch.load(f'./db/text_vecs/{self.split}/templ_lstm_emb.pt', map_location=cpu)
        self.no_txt_f = torch.zeros(256)

    self.img_size = img_size
    self.is_transform = is_transform
    self.mean = np.array([125.00, 125.00, 125.00]) # average intensity

    self.root = root
    self.dataPath = root + exp + '/' + self.split + '_impaths.txt'
    self.labelPath = root + exp + '/' + self.split + '_imclasses.txt'

    f_data = open(self.dataPath,'r')
    f_label = open(self.labelPath,'r')
    data_lines = f_data.readlines()
    label_lines = f_label.readlines()

    for i in range(len(data_lines)):
      self.inputs.append(root+data_lines[i][0:-1])
      self.targets.append(int(label_lines[i].split()[0])) # label: [road class, wet/dry, video index]
    
    classnamesPath = root + exp + '/' + self.split + '_classnames.txt'
    f_classnames = open(classnamesPath, 'r')
    data_lines = f_classnames.readlines()
    for i in range(len(data_lines)):
        self.class_names.append(data_lines[i][0:-1])

    assert(self.n_classes == len(self.class_names))

    print('%s %s %d classes'%(split, self.split, len(self.class_names)))
    print('Load %s: %s: %d samples'%(split, self.split,  len(self.targets)))

    if is_supcon:
        self.transform = SupConTwoCropTransform()
    else:
        self.transform = Transform(self.img_size)
    self.template_transform = transforms.Compose([
      transforms.Resize(self.img_size),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, index):
    img_path = self.inputs[index]
    gt = self.targets[index]
    gt = torch.ones(1).type(torch.LongTensor)*gt
    if self.split == 'test':
        pdb.set_trace()

    # Load images and templates. perform augmentations
    img = Image.open(img_path)
    template = Image.open(self.root + self.split + '/template_ordered/%02d.jpg' % (gt))

    if random.random() < self.proto_rate:
        img = template

    gt = gt-1

    if self.use_txt:
        jpg_txt = self.jpg_txt_features.get(img_path, self.no_txt_f)
        templ_txt = self.templ_txt_features.get(self.root + self.split + '/template_ordered/%02d.jpg' % (gt), self.no_txt_f)
    
    if self.split == 'belga':
        views = self.transform(img)
        templ_views = self.transform(template)

        res = {
            'view1': views[0],
            'view2': views[1],
            'templ1': templ_views[0],
            'templ2': templ_views[1],
            'label': gt,
        }

        if self.use_txt:
            res.update({
                'jpg_txt': torch.tensor(jpg_txt).float(),
                'templ_txt': torch.tensor(templ_txt).float()
            })

        return res
    else:
        if self.use_txt:
            return {
                'img': self.template_transform(img),
                'template': self.template_transform(template),
                'label': gt,
                'jpg_txt': torch.tensor(jpg_txt).float(),
                'templ_txt': torch.tensor(templ_txt).float()
            }

        return self.template_transform(img), gt, self.template_transform(template)

  def load_template(self, target, augmentations=None):

    # if augmentation is not specified, use self.augmentations. Unless use input augmentation option.
    if augmentations is None:
        augmentations = self.template_transform
    img_paths = []
    txt = []
    
    for id in target:
        img_paths.append(self.root + self.split +'/template_ordered/%02d.jpg'%(id+1))

    target_img = []
    for img_path in img_paths:
        img = Image.open(img_path)

        img = augmentations(img)

        target_img.append(img)

        if self.use_txt:
            txt.append(torch.tensor(self.templ_txt_features.get(img_path, self.no_txt_f)).float())

    if self.use_txt:
        return torch.stack(target_img, dim=0), torch.stack(txt, dim=0)
    return torch.stack(target_img, dim=0)
    




