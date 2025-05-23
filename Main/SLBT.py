import cv2
import numpy as np


def slbt(img1):

    def lbp(img):
        def assign_bit(picture, x, y, c):  # comparing bit with threshold value of centre pixel
            bit = 0
            try:
                if picture[x][y] >= c:
                    bit = 1
            except:
                pass
            return bit

        def local_bin_val(picture, x, y):  # calculating local binary pattern value of a pixel
            eight_bit_binary = []
            centre = picture[x][y]
            powers = [1, 2, 4, 8, 16, 32, 64, 128]
            decimal_val = 0
            # starting from top right,assigning bit to pixels clockwise
            eight_bit_binary.append(assign_bit(picture, x - 1, y + 1, centre))
            eight_bit_binary.append(assign_bit(picture, x, y + 1, centre))
            eight_bit_binary.append(assign_bit(picture, x + 1, y + 1, centre))
            eight_bit_binary.append(assign_bit(picture, x + 1, y, centre))
            eight_bit_binary.append(assign_bit(picture, x + 1, y - 1, centre))
            eight_bit_binary.append(assign_bit(picture, x, y - 1, centre))
            eight_bit_binary.append(assign_bit(picture, x - 1, y - 1, centre))
            eight_bit_binary.append(assign_bit(picture, x - 1, y, centre))
            # calculating decimal value of the 8-bit binary number
            for i in range(len(eight_bit_binary)):
                decimal_val += eight_bit_binary[i] * powers[i]

            return decimal_val

        m, n, _ = img.shape
        gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting image to grayscale
        lbp_photo = np.zeros((m, n), np.uint8)
        # converting image to lbp
        for i in range(0, m):
            for j in range(0, n):
                lbp_photo[i, j] = local_bin_val(gray_scale, i, j)

        return lbp_photo

    l = lbp(img1)

    return l





