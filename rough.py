from PIL import Image
import imageio
import numpy as np
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt
# Import segmentation maps from imgaug
from imgaug.augmentables.segmaps import SegmentationMapOnImage


def augmented_image(image_pth, mask_pth):
    image = Image.open(image_pth)
    pil_mask = Image.open(mask_pth)
    np_msk = np.array(pil_mask)
    np_img = np.array(image)
    flipped_msk = np.fliplr(np_msk)
    flipped_img = np.fliplr(np_img)
    img = Image.fromarray(np.asarray(flipped_img.astype('uint8')), 'RGB')
    mask = Image.fromarray(np.asarray(flipped_msk.astype('uint8')), 'RGB')
    # mask.show()
    # # plt.imshow(img)
    # # Convert mask to binary map
    # np_mask = np.array(pil_mask)
    # # Create segmentation map
    # segmap = np.zeros(image.shape, dtype=bool)
    # segmap[:] = np_mask
    # segmap = SegmentationMapOnImage(segmap, shape=image.shape)
    # plt.imshow(np.asarray(segmap.draw()[0]))
    # plt.show()
    # seq = iaa.Sequential([
    #     # iaa.CoarseDropout(0.1, size_percent=0.2), # coarse dropout
    #     iaa.Affine(rotate=(-20, 20)),  # rotate the image
    #     # iaa.ElasticTransformation(alpha=10, sigma=1)  # elastic transform
    # ])
    # image_aug, segmap_aug = seq(image=image, segmentation_maps=segmap)
    # mask = Image.fromarray(np.asarray(segmap_aug.draw()[0].astype('uint8')), 'RGB')
    # im = Image.fromarray(np.asarray(image_aug.astype('uint8')), 'RGB')
    return img, mask


if __name__ == '__main__':

    augmented_image("images/img/train1.jpg", 'images/mask/train1_mask.jpg')

# image = imageio.imread("images/img/train1.jpg")
# # Open image with mask
# pil_mask = Image.open('images/mask/train1_mask.jpg')
# # pil_mask = pil_mask.convert("1")
# # Convert mask to binary map
# np_mask = np.array(pil_mask)
# # np_mask = np.clip(np_mask, 0, 1)
#
# # Create segmentation map
# segmap = np.zeros(image.shape, dtype=bool)
# segmap[:] = np_mask
# segmap = SegmentationMapOnImage(segmap, shape=image.shape)
#
# # Initialize augmentations pipeline
# seq = iaa.Sequential([
#     # iaa.CoarseDropout(0.1, size_percent=0.2), # coarse dropout
#     iaa.Affine(rotate=(-50, 50)), # rotate the image
#     iaa.ElasticTransformation(alpha=10, sigma=1) # elastic transform
# ])
#
# # Apply augmentations for image and mask
# image_aug, segmap_aug = seq(image=image, segmentation_maps=segmap)
# side_by_side = np.hstack([
#     np.asarray(segmap.draw_on_image(image)[0]), # show blend of original image and segmentation map
#     np.asarray(segmap_aug.draw_on_image(image_aug)[0]),  # show blend of augmented image and segmentation map
#     np.asarray(segmap_aug.draw()[0])  # show only the augmented segmentation map
# ])
#
# fig, ax = plt.subplots()
# ax.axis('off')
# plt.title('Augmentations for segmentation masks')
# ax.imshow(side_by_side)
print(1)