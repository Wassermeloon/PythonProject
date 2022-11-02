import torch
from torch import nn, Tensor
import math
import torchvision
import torch.nn.functional as f
from typing import List, Dict, Tuple, Optional
from .image_list import ImageList



class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transform before feeding the data to a GeneralizedRCNN
    model

    The transforms it performs:
        - input normalization (mean subtraction and std division)
        - input / target resizing to mach the min_size and the max size
        - must considerate the bonding box

    """
    def __init__(self,
                 min_size: int,
                 max_size: int,
                 image_mean: float,
                 image_std: float):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, tuple):
            min_size = (min_size,)

        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.dtype = None
        self.device = None

    def normalize(self, image):
        """ normalization process"""
        self.dtype = image.dtype
        self.device = image.device
        mean = torch.as_tensor(self.image_mean, dtype=self.dtype, device=self.device)
        std = torch.as_tensor(self.image_std, dtype=self.dtype,device=self.device)
        normal_image = (image - mean[:, None, None]) - std[:, None, None]
        return normal_image

    def torch_choice(self, k):
        """

        :param k: the candidate of given size
        :return: the randomly choosed ons size
        """
        index = int(torch.empty(1).uniform_(0, len(k)).item())
        return k[index]

    def resize(self, target, image):

        # type: (Optional[Dict[str, Tensor]], Tensor) ->(Optional[Dict[str, Tensor]], Tensor)
        """
        the goal is resize the image between the max_size and min_size with the bonding box info
        :param target: it is a dictionary
        :param image: it is a tensor
        :return: target: bonding box after resize operation, it is also a dictionary
                image: image with after resize operation, it is a tensor
       """
        # image shape is [channel, height, width]
        h, w = image.shape[-2:]
        im_shape = torch.as_tensor(image.shape[-2:], device=self.device, dtype=self.dtype)
        img_min_size = float(torch.min(im_shape))
        img_max_size = float(torch.max(im_shape))

        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            size = float(self.min_size[-1])

        scale_ratio = size / img_min_size

        if scale_ratio * img_max_size > self.max_size:
            scale_ratio = self.max_size / img_max_size

        # scale image: interpolation -> bilinear
        # bilinear support only 4d : [C, H, W] -> [N, C, H, W]
        image = f.interpolate(image[None,:,:,:], scale_factor=scale_ratio,
                              mode="bilinear", align_corners=False)[0]

        if target is None:
            return  target, image
        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        return target, image


    def batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> int
        """

        :param images: input images, it is a list of many images
        :param size_divisible:
        :return: batch_images

        """
        if torchvision._is_tracing():
            return self._onnx_batch_images(images, size_divisible)

        # get the max height und width of all images
        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)

        # adjust height
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        # [batch, channel, height, width]
        batch_shape = [len(images)] + max_size

        # create tensor with batch_shape and the elements are all 0
        batch_images = images[0].new_full(batch_shape, 0)

        for index, img in enumerate(images):
            batch_images[index, : img.shape[0], : img.shape[1] : img.shape[2]] = img

        return batch_images


    def max_by_axis(self, images_shape):
        # type: (List[List[int]]) -> List[int]
        maxes = images_shape[0]
        for sublist in images_shape[1:]:
            for index, item in sublist:
                maxes[index] = max(maxes[index], item)

        return maxes

    def forward(self, images, targets=None):
        # get a list of images, because we need the original information, so we copy it
        images = [image for image in images]

        for i in range(len(images)):
            image = images[i]
            targets_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError("the image dimension must be 3, ranged by [channel, height, width]")

            image = self.normalize(image)
            targets_index, image = self.resize(targets_index, image)
            images[i] = image
            if targets is not None and targets_index is not None:
                targets[i] = targets_index

        # save the size of images after resize
        images_size = [im.shape[-2:] for im in images]
        images = self.batch_images(images)
        image_size_list = torch.jit.annotate(List[Tuple[int, int]], [])

        for image_size in images_size:
            assert len(image_size) == 2
            image_size_list.append((image_size[0], images_size[1]))

        image_list = ImageList(images, image_size_list) # record the size after resize

        return image_list, targets

    def postprocess(self,results, image_shapes, original_image_shape):
        # type: (List[Dict[str, Tensor]], List[Tuple[int, int]], List[Tuple[int, int]]) -> (List[Dict[str, Tensor]])
        """

        :param results: the predicted results, it is the predicted target information
        :param image_shapes: the predicted results image shape
        :param original_image_shape: the original image shapes
        :return: the predicted box which maps on the original images
        """
        if self.training:
            return results
        for i ,(pred, im_s, o_im_s) in enumerate(zip(results, image_shapes, original_image_shape)):
            bbox = pred["boxes"]
            bbox = resize_boxes(bbox, im_s, o_im_s)
            results[i]["boxes"] = bbox
        return results

def resize_boxes(boxes, original_size, new_size):
    """
    :param boxes: boxes information of the target
    :param original_size: the original size of img
    :param new_size: the size of img after interpolation
    :return: the boxes information after resize
    """
    rations = [
        torch.as_tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.as_tensor(s_org, dtype=torch.float32, device=boxes.device)
        for s, s_org in zip(new_size, original_size)
    ]

    rations_height = rations[0]
    rations_width = rations[1]
    x_min = boxes[:, 0] * rations_width
    x_max = boxes[:, 2] * rations_width
    y_min = boxes[:, 1] * rations_height
    y_max = boxes[:, 3] * rations_height

    return torch.stack((x_min, y_min, x_max, y_max), dim=1)
