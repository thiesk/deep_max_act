import torch.nn
from torch import nn
from torchvision import models
import numpy as np
from torch.autograd import Variable
import parameters
import cv2 as cv
from tqdm import tqdm

fake_img = Variable(torch.zeros(parameters.size), requires_grad = True)
model = models.vit_b_32(pretrained = True)
model.eval()

step = 10000
classish = 4
optimizer = torch.optim.Adam([fake_img], lr=0.06)

for i in tqdm(range(step)):
    out = model(fake_img)
    output_tensor = torch.zeros_like(out)
    # Set the value at index i to the original value
    output_tensor[0][classish] = out[0][classish]
    target_score = out[0, classish]

    # Backward pass
    optimizer.zero_grad()
    target_score.backward()

    optimizer.step()

    fake_img.data = torch.clamp(fake_img.data, 0, 1)

    if i % 10 == 0:

        img = np.array(fake_img.detach().numpy())
        img = img.transpose(0, 2, 3, 1)
        img = np.squeeze(img, axis=0) * 255
        cv.imwrite(f"output/{i}lmao.png", img)
