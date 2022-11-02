import torch
from torch.jit.annotations import List, Tuple
from torch import Tensor

class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size and
    storing in a field the original sizes of each image
    """

    def __init__(self, images, image_size_list):
        # type: (Tensor, List[Tuple[int, int]])
        """

        :param images: image tensors in a list
        :param image_size_list: the size of images after resize, we always know the original size

        """
        self.images = images
        self.image_size_list = image_size_list
    def to(self,device):
        # type: (Device)
        cast_tensor = self.images.to(device)

        return ImageList((cast_tensor, self.image_size_list))