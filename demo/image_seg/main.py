import PIL
import numpy as np
import torch
from torchvision import transforms

from seg_model import ImageSeg


if __name__ == '__main__':
    img = np.array(PIL.Image.open("img.png"))

    # print(ds_test.images[1])
    # img = np.array(ds_test[1][0])
    imgH = img.shape[0]
    imgW = img.shape[1]
    model = ImageSeg()
    seg = model.process(img, target_num=40)
    seg = torch.tensor(seg).long()


    import matplotlib.pyplot as plt
    print(seg.max() + 1)
    # print(si_value)
    torch.random.manual_seed(321)
    imgc = torch.rand((seg.max() + 1, 3))
    imgc2 = torch.rand((seg.max() + 1, 3))
    imgs = imgc[seg].reshape(imgH, imgW, 3)
    imgs2 = transforms.ToTensor()(img).permute(1, 2, 0).reshape(imgH, imgW, 3)
    imgs = (imgs2 + imgs) * 0.5

    imgcl = seg.reshape(1, 1, imgH, imgW).float()
    kernel1 = torch.Tensor([-1, 1, 0]).reshape(1, 1, 1, 3)
    kernel2 = torch.Tensor([-1, 1, 0]).reshape(1, 1, 3, 1)
    import torch.nn.functional as F
    imgcl = torch.max(
        torch.abs(F.conv2d(imgcl, kernel1, padding=(0, 1))) + torch.abs(F.conv2d(imgcl, kernel2, padding=(1, 0))), 1)

    imgs[:, :, 0] += imgcl[0][0]
    imgs[:, :, 1] -= imgcl[0][0]
    imgs[:, :, 2] -= imgcl[0][0]

    imgs = imgs.clamp(max=1, min=0)
    imgs = np.array(imgs)
    plt.imsave("output.png", imgs)
    plt.axis("off")
    plt.imshow(imgs)
    plt.show()

