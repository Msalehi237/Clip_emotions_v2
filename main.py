import torch
import clip
from data_loader import ImageDataSet
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import PIL
import os
from vizualization import sample_visualizer, top_k_visualizer, class_visualizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

#root_data = r'C:\Users\msalehi\Desktop\Projects\Image emotion project\Data\IndieGoGo\data1\images'
root_data = 'data'

data = ImageDataSet(root_data, transform=preprocess)
loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False, drop_last=False)

text = clip.tokenize(["this is a sketch of a logo",
                      "this is a graphically design",
                      "this is a real photograph taken with camera"
                      ]).to(device)

with torch.no_grad():
    image_index = []
    all_probs = []
    logits = []
    classes = []
    continuous = []
    for images, idx in tqdm(loader):
        image_index.append(idx)
        logits_per_image, logits_per_text = model(images, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        all_probs.append(probs)
        logits.append(logits_per_image)
        # print(idx, logits)
        for i in range(len(probs)):
            # happy.append(probs[i][0])
            classes.append(probs[i].argmax())
            continuous.append(np.sum(np.array([1, 2, 3]) * np.array(probs[i])))

image_index = [item for tpl in image_index for item in tpl]
all_probs = [item for tpl in all_probs for item in tpl]

# Putting together extracted values
authenticity_per_image = pd.DataFrame({'image': image_index, 'Authenticity_class': classes,
                                       'Authenticity_value': continuous})
authenticity_per_image.to_csv(r'C:\Users\msalehi\PycharmProjects\ImageEmotion\Clip_emotions_v2\Authenticity.csv')

## Visualization
# sample_visualizer(results_per_image, root_data, n=40)
# top_k_visualizer(results_per_image, root_data, k=40)
# class_visualizer(results_per_image, root_data, n=75)
