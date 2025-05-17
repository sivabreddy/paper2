import cv2, numpy as np
from timm.data.random_erasing import RandomErasing
from torchvision import transforms
import  matplotlib.pyplot as plt

def Augmentation(input_im):

    ################ rotating #################
    rows, cols, dim = input_im.shape
    angle,angle = np.radians(30),np.radians(10)
    # transformation matrix for Rotation
    M = np.float32([[np.cos(angle), -(np.sin(angle)), 0],[np.sin(angle), np.cos(angle), 0],[0, 0, 1]])
    rotated_img = cv2.warpPerspective(input_im, M, (int(cols), int(rows)))
    rotated_img = cv2.resize(rotated_img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    height, width = input_im.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 50, .5)
    #rotated_img = rotated_img[:,:,0]

    ############### cropping ##################
    cropped_image = input_im[int(cols*0.3): , int(cols*0.3):] #30%
    cropped_image = cv2.resize(cropped_image, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    #cropped_image = cropped_image[:,:,0]

    ############### flipping ##################

    fliped_img = cv2.flip(input_im, 0)

    ############### random Erasing ############

    ## random erasing ##

    grayscale = cv2.cvtColor(input_im, cv2.COLOR_BGR2GRAY)
    x = transforms.ToTensor()(grayscale)
    # plt.imshow(x.permute(1, 2, 0))

    random_erase = RandomErasing(probability=1, mode='pixel', device='cpu')
    r_e = random_erase(x).permute(1, 2, 0)
    r_e1 = (r_e).numpy()
    r_e1 = r_e1.reshape(r_e1.shape[0], r_e1.shape[1])
    r_e1_1 = (np.dstack((r_e1, r_e1, r_e1)) * 255.999).astype(np.uint8)
    # r_e1_1 = r_e1 * 255
    # r_e1_1 = np.uint8(r_e1_1)

    return  rotated_img, cropped_image,fliped_img,r_e1_1