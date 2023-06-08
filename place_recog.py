import clip
import torch
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import time




def main():
    imgs_dir = '/home/padfoot7/Desktop/RRC/CLIP_Project/lidar_lseg/data/lego_loam_map2/lego_loam_images_png/'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px")
    cosine_similarity = torch.nn.CosineSimilarity(dim=1)



    image = preprocess(Image.open("data/topomap/1.jpeg")).unsqueeze(0).to(device)
    feats_query_img = model.encode_image(image).cpu().to(torch.float32)
    
    feats_ref_imgs = []

    idx_start, idx_end = 1455, 1541
    with torch.no_grad():
        for i in range(idx_start, idx_end):
            tic = time.time()
            print(i)
            refimgage = preprocess(Image.open(f"{imgs_dir}{str(i).zfill(6)}.png")).unsqueeze(0).to(device)
            feats_ref_img = model.encode_image(refimgage).cpu().to(torch.float32)
            # feats_ref_img = model.encode_image(refimgage)
            feats_ref_imgs.append(feats_ref_img)
            toc = time.time()
            print('Time: ', toc-tic)


    feats_ref_imgs = torch.vstack(feats_ref_imgs)
    sim = cosine_similarity(feats_query_img, feats_ref_imgs)
    sim = sim.detach().numpy()
    idxs = np.arange(len(sim))+idx_start
    plt.plot(idxs, sim); plt.show()
    breakpoint()

def test_lang():
    imgs_dir = '/home/padfoot7/Desktop/RRC/CLIP_Project/lidar_lseg/data/lego_loam_map2/lego_loam_images_png/'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px")
    cosine_similarity = torch.nn.CosineSimilarity(dim=1)


    text = clip.tokenize(["benches on the left of the road"]).to(device)
    feats_text = model.encode_text(text).cpu().to(torch.float32)
    
    feats_ref_imgs = []
    idx_start, idx_end = 1455, 1541
    with torch.no_grad():
        for i in range(idx_start, idx_end):
            print(i)
            refimgage = preprocess(Image.open(f"{imgs_dir}{str(i).zfill(6)}.png")).unsqueeze(0).to(device)
            feats_ref_img = model.encode_image(refimgage).cpu().to(torch.float32)
            feats_ref_imgs.append(feats_ref_img)

    feats_ref_imgs = torch.vstack(feats_ref_imgs)
    sim = cosine_similarity(feats_text, feats_ref_imgs)
    sim = sim.detach().numpy()
    idxs = np.arange(len(sim))+idx_start
    plt.plot(idxs, sim); plt.show()
    breakpoint()

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px")
    cosine_similarity = torch.nn.CosineSimilarity(dim=1)



    image = preprocess(Image.open("data/topomap/4.jpeg")).unsqueeze(0).to(device)
    feats_query_img = model.encode_image(image).cpu().to(torch.float32)
    
    feats_ref_imgs = []
    with torch.no_grad():
        for i in range(1,12):
            print(i)
            refimgage = preprocess(Image.open(f"data/topomap/{i}.jpeg")).unsqueeze(0).to(device)
            feats_ref_img = model.encode_image(refimgage).cpu().to(torch.float32)
            feats_ref_imgs.append(feats_ref_img)

    feats_ref_imgs = torch.vstack(feats_ref_imgs)
    sim = cosine_similarity(feats_query_img, feats_ref_imgs)
    breakpoint()



    pass


if __name__=='__main__':
    main()