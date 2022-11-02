import random
from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transfoms =transforms

    def __call__(self, target, image):
        for t in self.transfoms:
            target, image = t(target, image)

        return target, image


class ToTensor(object):
    def __call__(self, target, image):
        image = F.to_tensor(image)
        return target, image


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, target, image):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            xx_min = width - bbox[:, 0]
            xx_max = width - bbox[:, 2]
            bbox[:, 0] = xx_max
            bbox[:, 2] = xx_min
            target["boxes"] = bbox
        return target, image

