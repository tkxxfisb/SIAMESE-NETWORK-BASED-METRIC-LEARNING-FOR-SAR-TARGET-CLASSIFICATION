import torchvision.transforms as T
from .gaussian_blur import GaussianBlur
T.GaussianBlur = GaussianBlur

imagenet_mean_std = [[0.5, ], [0.5, ]]

class SiameseTransform():
    def __init__(self, mean_std=imagenet_mean_std):
        image_size = 32
        p_blur = 0.5

        self.transform = T.Compose([
            # T.RandomResizedCrop(image_size, scale=(0.2, 0.1)),
            T.Resize([image_size, image_size]),
            T.RandomRotation(degrees=(0, 360)),
            T.RandomHorizontalFlip(),
            # T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            # T.RandomGrayscale(p=0.2),
            # T.RandomApply([T.GaussianBlur(kernel_size=image_size //20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])

    def __call__(self, x0, x1):
        x0 = self.transform(x0)
        x1 = self.transform(x1)
        return x0, x1
