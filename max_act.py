import torch.nn
from torch import nn
from torchvision import models
import numpy as np
from torch.autograd import Variable
import parameters
import cv2 as cv
from tqdm import tqdm
import torch.nn.functional as F
from video import bake_vid
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
for file in os.listdir("./output/"):
    os.remove(f"./output/{file}")

fake_img = Variable(torch.zeros(parameters.size, device=device), requires_grad = True)
model_dict = {"google":models.googlenet(pretrained = True).to(device),
              "resnet18":models.vit_b_32(pretrained = True).to(device)}

model = model_dict[parameters.model]
model.eval()

step = parameters.step
classish = parameters.vis_class
optimizer = torch.optim.Adam([fake_img], lr=parameters.lr)

for i in tqdm(range(step)):
    out = model(fake_img)
    output_tensor = torch.zeros_like(out)
    # Set the value at index i to the original value
    output_tensor[0][classish] = out[0][classish]
    target_score = out[0, classish]

    # Backward pass
    optimizer.zero_grad()
    target_score.backward()

    # Apply total variation regularization
    tv_loss = total_variation_regularization_loss(fake_img, parameters.lambda_reg)
    tv_loss.backward()

    optimizer.step()

    fake_img.data = torch.clamp(fake_img.data, 0, 1)

    if i % 5 == 0:

        img = np.array(fake_img.cpu().detach().numpy())
        img = img.transpose(0, 2, 3, 1)
        img = np.squeeze(img, axis=0) * 255
        cv.imwrite(f"output/{i}.png", img)

name = f"./videos/video_lam{parameters.lambda_reg}_lr{parameters.lr}_c{parameters.vis_class}"
bake_vid(name)