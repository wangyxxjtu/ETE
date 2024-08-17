from PIL import Image
import torch
from PIL import Image
from torch import nn, optim
import glob
import os
import pandas as pd
import json
import numpy as np
import clip
from torch.utils.data import Dataset, DataLoader, BatchSampler
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import random
import collections
import torch.utils.data as data
import json
import torch.optim as optim
import numpy as np
import random
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from util import *
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import torch
from tqdm import tqdm
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import argparse
from pudb.remote import set_trace
import pdb

parser = argparse.ArgumentParser()   
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
parser.add_argument('--dataset', default='N4News', type=str, help='the training dataset')    
parser.add_argument('--suffix', default='', type=str, help='suffix for checkpoint saving')    
parser.add_argument('--output', default='checkpoints', type=str, help='the dir to save the trained checkpoints')    
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--expert_config', default='444', type=str, help='Configuration of expert number. For example: 424 to initialize the number is 4(vis),2(share),4(text) ')    
parser.add_argument('--backbone', default='ViT-B/32', type=str, help='the clip vision backbone (ViT-B/32, ViT-B/16, ViT-L/14, RN101, RN50x4)')    
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--batch_size', default=192, type=int)
parser.add_argument('--epoch', default=20, type=int)
parser.add_argument('--beta', default=0.1, type=float)
parser.add_argument('--lr', default=1e-5, type=float)
args = parser.parse_args()

seed = args.seed
BATCH_SIZE=args.batch_size
EPOCH = args.epoch
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(args.backbone, device=device, jit=False, expert_config=args.expert_config)

if not os.path.exists(args.output):
    os.makedirs(args.output, exist_ok=True)

class N4News(data.Dataset):
    def __init__(self, data_path=None, data_split='test', transform = None, shape=(256,256)):
        self.data_info = json.load(open(f'{data_path}news/nytimes_{data_split}.json'))
        self.cap_info = json.load(open(f'{data_path}news/CapDsc_mistral_{data_split}.json'))
        self.shape = shape
        self.transform = transform
        self.data_root = data_path

    def __getitem__(self, index):
        im_id = self.data_info[index]['image_id']
        im_path = os.path.join(self.data_root, f'imgs/{im_id}.jpg' )

        #im = cv2.imread(im_path, cv2.COLOR_BGR2RGB)
        im = Image.open(im_path).convert('RGB')
        im = preprocess(im)

        text = self.data_info[index]['caption']
        text = clip.tokenize(text, truncate=True)

        cap = self.cap_info[os.path.basename(im_path)]
        cap = clip.tokenize(cap, truncate=True)
        #return {'image': im, 'text': text, 'im_path': im_path}
        return im, text[0], cap[0]

    def __len__(self):
        #return  len(self.data_info)
        return 512

#===============================================================================================================================
class VisualNews(Dataset):
    def __init__(self, image_root='', data_split='test', transform = None, shape=(256,256)):
        self.annotation = json.load(open(f'{image_root}/{data_split}.json'))
        self.cap_info = json.load(open(f'{image_root}/CapDsc_mistral_{data_split}.json'))
        self.transform = transform
        self.image_root = image_root
        self.shape = shape
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        img_id = self.annotation[index]['image_path'][1:]
        
        image_path = os.path.join(self.image_root, f'images/{img_id}')        
        image = Image.open(image_path).convert('RGB')   
        image = preprocess(image)
        
        #caption = pre_caption(self.annotation[index]['headline'], self.max_words) 
        caption = self.annotation[index]['caption']

        text = clip.tokenize(caption, truncate=True)
        cap = self.cap_info[image_path.replace('/hfut/', '/star/')]
        cap = clip.tokenize(cap, truncate=True)
        
        return image, text[0], cap[0]
 
#==============================================================================================================================
class GoodNews(Dataset):
    def __init__(self, image_root='', data_split='test', transform = None, shape=(256,256)):
        self.data_info = json.load(open(f'{image_root}/new_goodnews_capim_{data_split}.json'))
        self.cap_info = json.load(open(f'{image_root}/train_cap_individual/New_Corr_CapDsc_mistral_{data_split}.json'))
        broken_images = open(f'{image_root}/broken_images.txt').readlines()

        self.transform = transform
        self.image_root = image_root
        self.shape = shape
        
        self.text = []
        self.images = []

        for item in tqdm(self.data_info):
            #text = item['headline']
           #im_id = item['image_path']
            #im_path = os.path.join(self.data_root, f'imgs/{im_id}.jpg' )
            img_id = item['image_id']
            if img_id+'\n' in broken_images or img_id not in self.cap_info:
                continue

            #if img_id == '5312524f38f0d81c811a1bf7_6.jpg':
            #    pdb.set_trace()

            im_path = os.path.join(image_root, f'images/{img_id}' )
            if not os.path.exists(im_path):
                continue
    
            self.images.append(im_path)
            text = item['captioning']
            cap = text.replace('\r\n', ' ')
            cap = cap.replace('\r', '')
    
            caption = cap         
            self.text.append(caption)

        print('Total items:', len(self.images))
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):    
        
        image_path = self.images[index]
        
        #image_path = os.path.join(self.image_root, f'images/{img_id}')        
        image = Image.open(image_path).convert('RGB')   
        image = preprocess(image)
        
        #caption = pre_caption(self.annotation[index]['headline'], self.max_words) 
        caption = self.text[index]
        cap = self.cap_info[os.path.basename(image_path)]

        text = clip.tokenize(caption, truncate=True)
        cap = clip.tokenize(cap,truncate=True)
        
        return image, text[0], cap[0]

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        if p.grad is not None:
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float() 

if device == "cpu":
    model.float()

if 'N4News' in args.dataset:
    train_dataset = N4News(data_path = args.dataset, data_split='train')
    test_dataset = N4News(data_path = args.dataset, data_split='dev')
    dataset = 'N4News'
elif 'GoodNews' in args.dataset:
    train_dataset = GoodNews(image_root = args.dataset, data_split='train')
    test_dataset = GoodNews(image_root = args.dataset, data_split='val')
    dataset ='GoodNews'
else:
    train_dataset = VisualNews(image_root = args.dataset, data_split='train')
    test_dataset = VisualNews(image_root = args.dataset, data_split='val')
    dataset = 'VisualNews'
#===================================================================================================
num_tasks = 0
train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4) #Define your own dataloader
#==================================================================================================

test_dataloader = DataLoader(test_dataset,batch_size = BATCH_SIZE*2, num_workers=4) #Define your own dataloader

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
#GoodNews 5e-6
#N4News 1e-5
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*EPOCH)

best_te_loss = 1e5
best_r = 0.
best_ep = -1
epoch_len = len(train_dataloader)

for epoch in range(EPOCH):
    step = 0
    tr_loss = 0
    model.train()
    for idx, batch in enumerate(train_dataloader):
        step += 1
        optimizer.zero_grad()
        images,texts, caps = batch 
        
        images= images.to(device, non_blocking=True)
        texts = texts.to(device, non_blocking=True)
        caps = caps.to(device, non_blocking=True)
    
        main_logits_per_image, main_logits_per_text, logits_per_image, logits_per_text, itm_loss = model(images, texts, caps, num_tasks)
        ground_truth = torch.arange(len(logits_per_image),dtype=torch.long,device=device)

        main_loss = (loss_img(main_logits_per_image,ground_truth) + loss_txt(main_logits_per_text,ground_truth))/2
        auxi_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        total_loss = main_loss + auxi_loss * args.beta + 0.1 * itm_loss

        total_loss.backward()
        #set_trace()
        #pdb.set_trace()

        convert_models_to_fp32(model)
        optimizer.step()
        scheduler.step()
        clip.model.convert_weights(model)

        tr_loss += total_loss.item()

        ratio = 100. * float(idx+1) / epoch_len
        print(f'[{ratio:.2f}%/Epoch:{epoch}] loss: {total_loss.item():.4f}, Best_epoch: {best_ep}, Best test Loss: {best_te_loss:.4f}')

    tr_loss /= step
    
    step = 0
    te_loss = 0
    with torch.no_grad():
        model.eval()
        for idx, batch in enumerate(tqdm(test_dataloader)):
            step += 1
            images,texts, caps = batch 
            
            images= images.to(device)
            texts = texts.to(device)
            caps = caps.to(device)
            main_logits_per_image, main_logits_per_text, logits_per_image, logits_per_text, _  = model(images, texts, caps, num_tasks)

            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

            total_loss = (loss_img(main_logits_per_image,ground_truth) + loss_txt(main_logits_per_text,ground_truth))/2.
            te_loss += total_loss.item()
        te_loss /= step

    if te_loss < best_te_loss:
        best_te_loss = te_loss
        best_ep = epoch
        state_dict = model.state_dict()
        torch.save(state_dict, f"{args.output}/{dataset}_best_model_{args.expert_config}_beta{args.beta}{args.suffix}.pt")
    
    print(f"[Evaluation] epoch {epoch}, tr_loss {tr_loss}, te_loss {te_loss}\n")

    state_dict = model.state_dict()
    torch.save(state_dict, f"{args.output}/{dataset}_last_model_{args.expert_config}_beta{args.beta}{args.suffix}.pt")
