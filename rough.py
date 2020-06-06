from PIL import Image
import numpy as np
from scipy import ndimage


def augmented_image(image_pth, mask_pth):
    image = Image.open(image_pth)
    pil_mask = Image.open(mask_pth)
    i = ndimage.rotate(image, 45, reshape=True)
    # i = np.array(i)
    # i = Image.fromarray(np.asarray(i.astype('uint8')), 'RGB')
    np_msk = np.array(pil_mask)
    np_img = np.array(image)
    flipped_msk = np.fliplr(np_msk)
    flipped_img = np.fliplr(np_img)
    rotated_img = ndimage.rotate(np_img, 30, reshape=False)
    rotated_msk = ndimage.rotate(np_msk, 30, reshape=False)
    img = [Image.fromarray(np.asarray(flipped_img.astype('uint8')), 'RGB'),
           Image.fromarray(np.asarray(rotated_img.astype('uint8')), 'RGB')]
    mask = [Image.fromarray(np.asarray(flipped_msk.astype('uint8')), 'RGB'),
            Image.fromarray(np.asarray(rotated_msk.astype('uint8')), 'RGB')]

    return img, mask


if __name__ == '__main__':

    augmented_image("images/img/train1.jpg", 'images/mask/train1_mask.jpg')
