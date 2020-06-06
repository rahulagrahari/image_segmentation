from PIL import Image
import numpy as np



def augmented_image(image_pth, mask_pth):
    image = Image.open(image_pth)
    pil_mask = Image.open(mask_pth)
    np_msk = np.array(pil_mask)
    np_img = np.array(image)
    flipped_msk = np.fliplr(np_msk)
    flipped_img = np.fliplr(np_img)
    img = Image.fromarray(np.asarray(flipped_img.astype('uint8')), 'RGB')
    mask = Image.fromarray(np.asarray(flipped_msk.astype('uint8')), 'RGB')

    return img, mask


if __name__ == '__main__':

    augmented_image("images/img/train1.jpg", 'images/mask/train1_mask.jpg')
