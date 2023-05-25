import torch
import clip
from data_loader import ImageDataSet
import pandas as pd
import numpy as np
from tqdm import tqdm
from vizualization import sample_visualizer, top_k_visualizer, class_visualizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

root_data = 'data'
data = ImageDataSet(root_data, transform=preprocess)
loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False, drop_last=False)

text = clip.tokenize(["this is a close up photo",
                      "this is a medium close up photo",
                      "this is a mid shot photo",
                      "this is a wide shot photo",
                      "this is a very wide shot photo",
                      "this is a extreme wide shot photo"
                      ]).to(device)

with torch.no_grad():
    image_index = []
    all_probs = []
    logits = []
    happy = []
    for images, idx in tqdm(loader):
        image_index.append(idx)
        logits_per_image, logits_per_text = model(images, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        all_probs.append(probs)
        logits.append(logits_per_image)
        # print(idx, logits)
        for i in range(len(probs)):
            #happy.append(probs[i][0])
            #happy.append(probs[i].argmax())
            happy.append(np.sum(np.array([1, 2, 3, 4, 5, 6]) * np.array(probs[i])))

image_index = [item for tpl in image_index for item in tpl]
all_probs = [item for tpl in all_probs for item in tpl]

results_per_image = pd.DataFrame({'image': image_index, 'value': happy})


## Visualization
sample_visualizer(results_per_image, root_data, n=40)
#top_k_visualizer(results_per_image, root_data, k=40)
#class_visualizer(results_per_image, root_data, n=75)

