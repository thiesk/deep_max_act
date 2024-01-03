import os

import torch.nn
from torchvision import models
import numpy as np
from torch.autograd import Variable
import parameters
import matplotlib as plt
import cv2 as cv
from tqdm import tqdm
from video import bake_vid

def total_variation(image):
    dx = image[:, :, 1:] - image[:, :, :-1]
    dy = image[:, 1:, :] - image[:, :-1, :]
    dx_norm = torch.norm(dx)
    dy_norm = torch.norm(dy)
    return dx_norm + dy_norm

def total_variation_regularization_loss(image, lambda_tv):
    return lambda_tv * total_variation(image)

# clean the old folder
for file in os.listdir("./output/"):
    os.remove(f"./output/{file}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

fake_img = Variable(torch.zeros(parameters.size, device=device), requires_grad = True)
model_dict = {"google":models.googlenet(pretrained = True).to(device),
              "resnet18":models.vit_b_32(pretrained = True).to(device),
              "res50":models.resnet18(pretrained = True).to(device)}

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

    if i % parameters.videostep == 0:

        img = np.array(fake_img.cpu().detach().numpy())
        img = img.transpose(0, 2, 3, 1)
        img = np.squeeze(img, axis=0) * 255
        cv.imwrite(f"output/{i}.png", img)

name = f"./videos/mod{parameters.model}_lam{parameters.lambda_reg}_lr{parameters.lr}_c{parameters.vis_class}"
bake_vid(name)