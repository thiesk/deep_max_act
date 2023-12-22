import torch.nn
from torch import nn
from torchvision import models
import numpy as np
import parameters
class FakeImg(nn.Module):
    def __init__(self, shape):
        super(FakeImg, self).__init__()
        self.fake_img = torch.nn.Parameter(data=torch.randn(size=shape))

    def forward(self, x):
        return self.fake_img
    def laat_zien(self):
        pass
        #self.fake_img.num

def get_model(size):
    #friesenjung layers
    model = models.resnet18(pretrained = True)
    img = FakeImg(size)
    for name, params in model.named_parameters():
        params.requires_grad = False

    # add fake image layer infront of model
    mod_model = nn.Sequential(img, model)
    return mod_model


if __name__ == "__main__":
    model = get_model(parameters.size)
    inputimg = torch.randn(size=parameters.size).numpy()
    out = model(inputimg)
    print(torch.argmax(nn.functional.softmax(out)))

    print("lamoamsdfpsadfbg")