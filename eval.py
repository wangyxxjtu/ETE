import torch
import clip
from PIL import Image
import sys
from util import *
from tqdm import tqdm,trange
import argparse
from pudb.remote import set_trace
import pdb

parser = argparse.ArgumentParser()   
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
parser.add_argument('--dataset', default='N4News', type=str, help='the training dataset')    
parser.add_argument('--suffix', default='', type=str, help='using the best or last model')    
parser.add_argument('--backbone', default='ViT-B/32', type=str, help='the clip vision backbone (ViT-B/32, ViT-B/16, ViT-L/14, RN101, RN50x4)')    
parser.add_argument('--output', default='checkpoints', type=str, help='the dir to save the trained checkpoints')    
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--expert_config', default='444', type=str, help='Configuration of expert number. For example: 424 to initialize the number is 4(vis),2(share),4(text) ')    
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--beta', default=0.1, type=float)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(args.backbone, device=device, expert_config=args.expert_config)

if 'N4News' in args.dataset:
    dataset = 'N4News'
elif 'GoodNews' in args.dataset:
    dataset ='GoodNews'
else:
    dataset = 'VisualNews'
model.load_state_dict(torch.load(f"{args.output}/{dataset}_best_model_{args.expert_config}_beta{args.beta}{args.suffix}.pt"))

def evaluation(dataset):
    raw_info = read_dataset(dataset=args.dataset)
    im_feat = []
    txt_feat = []
    with torch.no_grad():
        for item in tqdm(raw_info):
            im_path, text, cap = item['image_path'], item['caption'], item['CapDsc']

            image = preprocess(Image.open(im_path)).unsqueeze(0).to(device)
            text = clip.tokenize([text], truncate=True).to(device)
            cap = clip.tokenize([cap], truncate=True).to(device)

            #image_fea = model.encode_image(image)
            #text_fea = model.encode_text(text)
            image_fea, text_fea = model.inference(image, text, cap)            
            #image_fea, text_fea = model.inference_image(image, cap), model.inference_text(text,cap)            

            im_feat.append(image_fea)
            txt_feat.append(text_fea)

        im_feat = torch.stack(im_feat, dim=0).squeeze()
        txt_feat = torch.stack(txt_feat, dim=0).squeeze()

        r1, r5, r10, r50, r100, medr, meanr = t2i_recall(txt_feat, im_feat)     
        t2i_results = f'Text 2 image Retrieval:\n R1: {r1}, R5: {r5}, R10: {r10}, R50: {r50}, R100: {r100}, medr: {medr}, meanr: {meanr}\n'

        print('Text 2 image retrieval:')
        print('Recall@1:', r1)
        print('Recall@5:', r5)
        print('Recall@10:', r10)
        print('Recall@50:', r50)
        print('Recall@100:', r100)
        print('Median Rank:', medr)
        print('Mean Rank:', meanr)

        r1, r5, r10, r50, r100, medr, meanr = t2i_recall(im_feat, txt_feat)     

        print('*'*80)
        print('Image 2 text retrieval:')
        print('Recall@1:', r1)
        print('Recall@5:', r5)
        print('Recall@10:', r10)
        print('Recall@50:', r50)
        print('Recall@100:', r100)
        print('Median Rank:', medr)
        print('Mean Rank:', meanr)

        i2t_results = f'Image 2 text Retrieval:\n R1: {r1}, R5: {r5}, R10: {r10}, R50: {r50}, R100: {r100}, medr: {medr}, meanr: {meanr}\n'

        with open(f'{args.output}/{dataset}_{args.expert_config}_{args.beta}.txt', 'w') as f:
            f.write(t2i_results)
            f.write(i2t_results)

def GoodNews():
    images, texts, capDes_info, text2img, img2text = read_dataset(dataset='GoodNews')
    im_feat = []
    txt_feat = []
    with torch.no_grad():
        for text_id, text in enumerate(tqdm(texts, total=len(texts))):
            im_ids = text2img[text_id]
            im_idx_id = ''.join(os.path.basename(images[im_ids[0]]).split('_')[:-1])
            cap = capDes_info[im_idx_id]
            cap = clip.tokenize([cap], truncate=True).to(device)

            text = clip.tokenize([text], truncate=True).to(device)
            text_fea = model.inference_text(text, text, False)
            txt_feat.append(text_fea)

            for im_id in im_ids:
                im_path = images[im_id]
                image = preprocess(Image.open(im_path)).unsqueeze(0).to(device)
                image_fea = model.inference_image(image, text, False)
                im_feat.append(image_fea)

        im_feat = torch.stack(im_feat, dim=0).squeeze()
        txt_feat = torch.stack(txt_feat, dim=0).squeeze()

        r1, r5, r10, r50, r100, medr, meanr = t2i_recall_GoodNews(txt_feat, im_feat, text2img)      

        print('Text 2 image retrieval:')
        print('Recall@1:', r1)
        print('Recall@5:', r5)
        print('Recall@10:', r10)
        print('Recall@50:', r50)
        print('Recall@100:', r100)
        print('Median Rank:', medr)
        print('Mean Rank:', meanr)

        r1, r5, r10, r50, r100, medr, meanr = t2i_recall_GoodNews(im_feat, txt_feat, img2text=img2text)     

        print('*'*80)
        print('Image 2 text retrieval:')
        print('Recall@1:', r1)
        print('Recall@5:', r5)
        print('Recall@10:', r10)
        print('Recall@50:', r50)
        print('Recall@100:', r100)
        print('Median Rank:', medr)
        print('Mean Rank:', meanr)

evaluation(dataset = args.dataset)
