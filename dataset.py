import math
import numpy as np
from PIL import Image
import torch.utils.data
import utils
import glob
import rough
import os


def full_image_loader(tile_size, img_path):
    dataset = tile_dataset(tile_size, img_path=[img_path])

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    return loader


def training_loader(batch_size, tile_size, shuffle=False):
    tile_stride_ratio = 0.5

    dataset = tile_dataset(tile_size, tile_stride_ratio=tile_stride_ratio, augumentation=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)

    return loader


def tile_dataset(tile_size, tile_stride_ratio=1.0, img_path=None, augumentation=False):
    # TODO: Perform data augmentation in this

    if img_path is None:
        img_ls = glob.glob(os.path.join("images", "img", "*.jpg"))
    else:
        img_ls = img_path
    mask_folder = os.path.join("images", "mask")
    x_tiles, y_tiles = None, None
    for i in img_ls:
        img_pth = i
        mask_pth = glob.glob(os.path.join(mask_folder, os.path.basename(i)[:os.path.basename(i).index(".")] + "*.jpg"))[0]
        x_image = utils.input_image(img_pth).convert('RGB')
        y_image = utils.label_image(mask_pth).convert('1')
        x_tile, y_tile = img_tile_generator(tile_size, x_image, y_image, tile_stride_ratio=tile_stride_ratio)
        if x_tiles is None and y_tiles is None:
            x_tiles, y_tiles = x_tile, y_tile
        else:
            x_tiles = np.vstack((x_tiles, x_tile))
            y_tiles = np.vstack((y_tiles, y_tile))

        if augumentation:
            x_aug, y_aug = rough.augmented_image(img_pth, mask_pth)
            x_tile_aug, y_tile_aug = img_tile_generator(tile_size, x_aug.convert('RGB'), y_aug.convert('1'),
                                                        tile_stride_ratio=tile_stride_ratio)
            x_tiles = np.vstack((x_tiles, x_tile_aug))
            y_tiles = np.vstack((y_tiles, y_tile_aug))

    # Clip tiles accumulators to the actual number of tiles
    # Since some tiles might have been discarded, n <= tile_count
    x_tiles = torch.from_numpy(x_tiles)
    y_tiles = torch.from_numpy(y_tiles)

    x_tiles = x_tiles.to(dtype=utils.x_dtype())
    y_tiles = y_tiles.to(dtype=utils.y_dtype())

    dataset = torch.utils.data.TensorDataset(x_tiles, y_tiles)

    return dataset


def img_tile_generator(tile_size, x_image, y_image, tile_stride_ratio=1.0):
    x_image = x_image
    y_image = y_image

    assert x_image.size == y_image.size

    tile_stride = (int(tile_size[0] * tile_stride_ratio),
                   int(tile_size[1] * tile_stride_ratio))

    tile_count, extended_size = utils.tiled_image_size(x_image.size, tile_size,
                                                       tile_stride_ratio)

    x_extended = utils.extend_image(x_image, extended_size, color=0)
    y_extended = utils.extend_image(y_image, extended_size, color=255)

    x_tiles = np.zeros((tile_count, 3, tile_size[0], tile_size[1]))
    y_tiles = np.zeros((tile_count, tile_size[0], tile_size[1]))

    def tile_generator():
        for x in range(0, extended_size[0], tile_stride[0]):
            for y in range(0, extended_size[1], tile_stride[1]):
                yield x, y, tile_size[0], tile_size[1]

    for n, (x, y, w, h) in enumerate(tile_generator()):
        box = (x, y, x + w, y + h)

        x_tile = np.array(x_extended.crop(box))
        y_tile = np.array(y_extended.crop(box))

        x_tiles[n, :, :, :] = np.moveaxis(x_tile, -1, 0)
        y_tiles[n, :, :] = y_tile

    return x_tiles[0:n, :, :, :], y_tiles[0:n,    :, :]
