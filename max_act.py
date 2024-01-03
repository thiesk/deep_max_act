import numpy as np
import torch
from torch.autograd import Variable
import parameters
import numbers
from torch.nn import functional as FF
import math
import cv2 as cv
from video import bake_vid
from torchvision import models
from tqdm import tqdm
import os

# Define the low-pass filter kernel
def low_pass_kernel(kernel_size, sigma):
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords -= (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()

def activation_max_with_gauss(model,
                              ):
    model.train()

    # clean the old folder
    for file in os.listdir("./output/"):
        os.remove(f"./output/{file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # generate fake image -- vielleicht mach andere init
    fake_img = Variable(torch.zeros(parameters.size, device=device), requires_grad=True)
    optim = torch.optim.SGD([fake_img], lr = parameters.lr)

    # create gradient of sigma sizes for different levels of detail
    sigmas = np.linspace(start=parameters.start_sigma, stop=parameters.end_sigma, num=parameters.step)

    # generation loop
    for i_iter in tqdm(range(0, parameters.step)):
        classish = parameters.vis_class
        # zero gradients
        model.zero_grad()

        # forward pass
        out = model(fake_img)

        # backward
        output_tensor = torch.zeros_like(out)
        # Set the value at index i to the original value
        output_tensor[0][classish] = out[0][classish]
        target_score = out[0, classish]
        target_score.backward()

        # get current sigma
        sigma = sigmas[i_iter]

        # gaussian blurr of image
        gradients = fake_img.grad.data
        for c in range(gradients.size(1)):
            # Low-pass filter kernel
            low_pass_filter = low_pass_kernel((parameters.kernel_size, parameters.kernel_size), sigma)
            low_pass_filter = low_pass_filter.view(1, 1, -1, 1)

            gradients[:, c, :, :] = FF.conv2d(gradients[:, c, :, :].unsqueeze(1),
                                              low_pass_filter.view(1, 1, parameters.kernel_size, parameters.kernel_size),
                                            padding=parameters.kernel_size // 2).squeeze(1)

        fake_img.data = fake_img.data + gradients * parameters.lr

        if i_iter % parameters.videostep == 0:
            img = np.array(fake_img.cpu().detach().numpy())
            img = img.transpose(0, 2, 3, 1)
            img = np.squeeze(img, axis=0) * 255
            cv.imwrite(f"output/{i_iter}.png", img)


# bla

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dict = {"google":models.googlenet(pretrained = True).to(device),
              "resnet18":models.vit_b_32(pretrained = True).to(device),
              "res50":models.resnet18(pretrained = True).to(device)}
model = model_dict[parameters.model]
activation_max_with_gauss(model=model, )
name = f"./videos/gradientblurr_mod{parameters.model}_lam{parameters.lambda_reg}_lr{parameters.lr}_c{parameters.vis_class}"
bake_vid(name)
