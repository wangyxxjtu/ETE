import numpy as np
import os
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import json
import glob
import pdb
from tqdm import tqdm

def cosine_similarity(x1, x2, logit_scale=1, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    #norm1 =  torch.norm(x1, 2, dim).clamp(min=eps).unsqueeze(1).repeat(1, x1.shape[1])
    #norm2 =  torch.norm(x2, 2, dim).clamp(min=eps).unsqueeze(1).repeat(1, x2.shape[1])

    #x1 = x1 / norm1 
    #x2 = x2 / norm2

    #sim = torch.matmul(x1, x2.t())

    x1 = F.normalize(x1)
    x2 = F.normalize(x2)
    sim = logit_scale * x1 @ x2.t()

    return sim

def t2i_recall_GoodNews(query, candidate, txt2img=None, img2text=None, return_ranks=False, sim_itc = None, only_itc=False, alpha=0.5):
    sims = cosine_similarity(query, candidate)
    sims = sims.cpu().numpy()

    #if only_itc:
    #    sims = sim_itc
    #else:
    #    sims = sims + alpha * sim_itc 
    npts = query.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
     
    if txt2img is None: #means the image as the query
        for index in range(npts):
            inds = np.argsort(sims[index])[::-1]
            ranks[index] = np.where(inds == img2text[index])[0][0]
            top1[index] = inds[0]
    else: #text as the query
        for index in range(npts):
            inds = np.argsort(sims[index])[::-1]
            rank = 1e20
            for idx in txt2img[index]:
                tmp = np.where(inds == idx)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
            top1[index] = inds[0]

    #comupte the recalls
    r1 = 100.0 * len(np.where(ranks<1)[0]) / (1.0 * len(ranks))
    r5 = 100.0 * len(np.where(ranks<5)[0]) / (1.0 * len(ranks))
    r10 = 100.0 * len(np.where(ranks<10)[0]) / (1.0 * len(ranks))
    r50 = 100.0 * len(np.where(ranks<50)[0]) / (1.0 * len(ranks))
    r100 = 100.0 * len(np.where(ranks<100)[0]) / (1.0 * len(ranks))

    medr = np.floor(np.median(ranks)) + 1

    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, r50, r100, medr, meanr)


def t2i_recall(text, image, logits_scale=1, return_ranks=False, sim_itc = None, only_itc=False, alpha=0.5):
    sims = cosine_similarity(text, image, logits_scale)
    sims = sims.cpu().numpy()

    #if only_itc:
    #    sims = sim_itc
    #else:
    #    sims = sims + alpha * sim_itc 

    npts = image.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        ranks[index] = np.where(inds == index)[0][0]
        top1[index] = inds[0]

    #comupte the recalls
    r1 = 100.0 * len(np.where(ranks<1)[0]) / (1.0 * len(ranks))
    r5 = 100.0 * len(np.where(ranks<5)[0]) / (1.0 * len(ranks))
    r10 = 100.0 * len(np.where(ranks<10)[0]) / (1.0 * len(ranks))
    r50 = 100.0 * len(np.where(ranks<50)[0]) / (1.0 * len(ranks))
    r100 = 100.0 * len(np.where(ranks<100)[0]) / (1.0 * len(ranks))

    medr = np.floor(np.median(ranks)) + 1

    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, r50, r100, medr, meanr)

class N4News_(data.Dataset):
    def __init__(self, data_path='', data_split='test', transform = None, shape=(256,256)):
        self.data_info = json.load(open(f'{data_path}news/nytimes_{data_split}.json'))
        self.shape = shape
        self.transform = transform
        self.data_root = data_path

    def __getitem__(self, index):
        im_id = self.data_info[index]['image_id']
        im_path = os.path.join(self.data_root, f'imgs/{im_id}.jpg' )

        #im = cv2.imread(im_path, cv2.COLOR_BGR2RGB)
        im = Image.open(im_path).convert('RGB')
        if self.transform is None:
            im = transform.ToTensor(im)
        else:
            im = self.transform(im)

        text = self.data_info[index]['headline']

        return {'image': im, 'text': text, 'im_path': im_path}

    def __len__(self):
        return len(self.data_info)

def get_dataloader(dataset='N4News', transform=None, shape=(256,256)):
    if dataset == 'N4News':
        dataset = N4News(transform=transform,shape=shape)

    elif dataset == 'VisualNews':
        pass

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, shuffle=False)

    return dataloader

def read_dataset(dataset='N4News',split ='test'):
    if 'N4News' in dataset:
        data_path = dataset
        data_info = json.load(open(f'{data_path}news/nytimes_{split}.json'))
        capDes_info = json.load(open(f'{data_path}news/CapDsc_mistral_{split}.json'))
        raw_info  = []
        for item in data_info:
            text = item['headline']
            cap = item['caption']
            img_id = item['image_id']
            im_path = os.path.join(data_path, f'imgs/{img_id}.jpg' )
            raw_info.append({'image_path':im_path, 'text':text, 'caption': cap, 'CapDsc':capDes_info[os.path.basename(im_path)], 'image_id': img_id})

        return raw_info
 
    elif dataset == 'GoodNews':
        data_path = dataset
        data_info = json.load(open(f'{data_path}/new_goodnews_capim_{split}.json'))
        cap_info = json.load(open(f'{data_path}/train_cap_individual/New_Corr_CapDsc_mistral_{split}.json'))

        raw_info  = []
        for item in data_info:
            text = item['title']
            cap = item['captioning'].replace('\r\n', ' ')
            cap = cap.replace('\r', '')
            
            img_id = item['image_id']
            im_path = os.path.join(data_path, f'images/{img_id}')
        
            cap_dsc = cap_info[os.path.basename(img_id)]
            raw_info.append({'image_path':im_path, 'text':text, 'caption': cap, 'image_id': img_id, 'CapDsc': cap_dsc})
        return raw_info

    elif dataset == 'VisualNews':
        data_path = dataset
        data_info = json.load(open(f'{data_path}/{split}.json'))
        capDes_info = json.load(open(f'{data_path}/CapDsc_mistral_{split}.json'))
        raw_info  = []
        #data_info = data_info[:100]
        for item in data_info:
            cap = item['caption']
            img_id = item['image_path'][1:]
            im_path = os.path.join(data_path, f'images/{img_id}' )
            #raw_info.append({'image_path':im_path,  'caption': cap, 'image_id': img_id})
            raw_info.append({'image_path':im_path, 'caption': cap, 'CapDsc':capDes_info[im_path.replace('/hfut/', '/star/')], 'image_id': img_id})

        return raw_info
