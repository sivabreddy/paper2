import glob,cv2,os
import re
from Main import Proposed_SegNet,Augmentation
import numpy as np
import lbp
import statistics
from skimage.filters.rank import entropy
from scipy.stats import kurtosis,skew
from scipy import ndimage
from skimage.morphology import disk
import SLBT

new_im_path='Output/roi'
new_fil_path='Output/amf'
new_seg_path='Output/segmented'
new_aug1_path='Output/rotation'
new_aug2_path='Output/cropping'
new_aug3_path = 'Output/flipping'
new_aug4_path = 'Output/random_erasing'
if not(os.path.exists('Output')):
    os.mkdir('Output')
if not(os.path.exists(new_im_path)):
    os.mkdir(new_im_path)

if not(os.path.exists(new_fil_path)):
    os.mkdir(new_fil_path)

if not(os.path.exists(new_seg_path)):
    os.mkdir(new_seg_path)
if not(os.path.exists(new_aug1_path)):
    os.mkdir(new_aug1_path)
if not(os.path.exists(new_aug2_path)):
    os.mkdir(new_aug2_path)
if not(os.path.exists(new_aug3_path)):
    os.mkdir(new_aug3_path)
if not(os.path.exists(new_aug4_path)):
    os.mkdir(new_aug4_path)

#### adaptive median filter ####
def calculate_median(array):
    """Return the median of 1-d array"""
    sorted_array = np.sort(array) #timsort (O(nlogn))
    median = sorted_array[len(array)//2]
    return median
def level_B(z_min, z_med, z_max, z_xy, S_xy, S_max):
    if(z_min < z_xy < z_max):
        return z_xy
    else:
        return z_med
def level_A(z_min, z_med, z_max, z_xy, S_xy, S_max):
    if(z_min < z_med < z_max):
        return level_B(z_min, z_med, z_max, z_xy, S_xy, S_max)
    else:
        S_xy += 2 #increase the size of S_xy to the next odd value.
        if(S_xy <= S_max): #repeat process
            return level_A(z_min, z_med, z_max, z_xy, S_xy, S_max)
        else:
            return z_med
def amf(image, initial_window, max_window):
    """runs the Adaptive Median Filter proess on an image"""
    xlength, ylength = image.shape  # get the shape of the image.

    z_min, z_med, z_max, z_xy = 0, 0, 0, 0
    S_max = max_window
    S_xy = initial_window  # dynamically to grow

    output_image = image.copy()

    for row in range(S_xy, xlength - S_xy - 1):
        for col in range(S_xy, ylength - S_xy - 1):
            filter_window = image[row - S_xy: row + S_xy + 1, col - S_xy: col + S_xy + 1]  # filter window
            target = filter_window.reshape(-1)  # make 1-dimensional
            z_min = np.min(target)  # min of intensity values
            z_max = np.max(target)  # max of intensity values
            z_med = calculate_median(target)  # median of intensity values
            z_xy = image[row, col]  # current intensity

            # Level A & B
            new_intensity = level_A(z_min, z_med, z_max, z_xy, S_xy, S_max)
            output_image[row, col] = new_intensity
    return output_image

def T2FCS(m5):
    def T2FCS_Filtering(image, row, col):
        currentElement = 0
        left = 0
        right = 0
        top = 0
        bottom = 0
        topLeft = 0
        topRight = 0
        bottomLeft = 0
        bottomRight = 0
        counter = 1
        currentElement = image[row][col]

        if not col - 1 < 0:
            left = image[row][col - 1]
            counter += 1
        if not col + 1 > width - 1:
            right = image[row][col + 1]
            counter += 1
        if not row - 1 < 0:
            top = image[row - 1][col]
            counter += 1
        if not row + 1 > height - 1:
            bottom = image[row + 1][col]
            counter += 1

        if not row - 1 < 0 and not col - 1 < 0:
            topLeft = image[row - 1][col - 1]
            counter += 1
        if not row - 1 < 0 and not col + 1 > width - 1:
            topRight = image[row - 1][col + 1]
            counter += 1
        if not row + 1 > height - 1 and not col - 1 < 0:
            bottomLeft = image[row + 1][col - 1]
            counter += 1
        if not row + 1 > height - 1 and not col + 1 > width - 1:
            bottomRight = image[row + 1][col + 1]
            counter += 1
        total = int(currentElement) + int(left) + int(right) + int(top) + int(bottom) + int(topLeft) + int(
            topRight) + int(bottomLeft) + int(bottomRight)
        avg = total / counter

        meau = avg
        n = 4

        Thr = 10
        T1 = np.arange(Thr - 2, Thr + 2)
        T2 = np.arange(Thr - 4, Thr + 4)
        T3 = np.arange(Thr - 8, Thr + 8)
        Fs = np.random.uniform(n - 1, n)

        # Case1#
        if meau in T1:
            I_x_y = currentElement
            # to find Y: absolute function of window based on neighbour pixel #
            Y = abs(I_x_y - np.mean(avg))
            # initially the value of za set to 0
            Za = 0
            Za = Za + I_x_y * Y
            # rounding process #
            Za = Za / 8

            # first value choosen as D
            D = Za
            if D > 10:
                Fij = 1 - (D - 1) / 4
            else:
                Fij = 1

            Inew = currentElement * Fij

        # Case2#
        elif meau in T2:
            I_x_y = currentElement
            # to find Y: absolute function of window based on neighbour pixel #
            Y = abs(I_x_y - np.mean(avg))
            # initially the value of za set to 0
            Za = 0
            Za = Za + I_x_y * Y
            # rounding process #
            Za = Za / 8

            # first value choosen as D
            D = Za
            if D > 10:
                Fij = 1 - (D - 1) / 4
            else:
                Fij = 1

            # new function generated as Fs
            n1 = 4
            Fs = np.sum(meau) / n1
            Inew = currentElement * (Fs / Fs)

        elif meau in T3:
            Inew = currentElement
        else:
            Inew = avg

        return Inew

    m5_tst = cv2.cvtColor(m5, cv2.COLOR_BGR2GRAY)
    height, width = m5.shape[0], m5.shape[1]
    img_t2fcs = np.zeros((height, width, 3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_t2fcs[i, j, 0] = T2FCS_Filtering(m5[:, :, 0], i, j)
            img_t2fcs[i, j, 1] = T2FCS_Filtering(m5[:, :, 1], i, j)
            img_t2fcs[i, j, 2] = T2FCS_Filtering(m5[:, :, 2], i, j)
    return img_t2fcs

def mark_seg_in_orgim(input,seg):
    input = input[:, :, 0]
    input = cv2.resize(input, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    input = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
    seg = cv2.resize(seg, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            if (seg[i][j] == 255):
                input[i + 1][j] = (0, 255, 255)
                input[i][j + 2] = (0, 255, 255)
    return input


def segment(input_im,org):
    input_im = cv2.resize(input_im, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    org = cv2.resize(org, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    org = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
    # Segmentation
    seg = Proposed_SegNet.Segnet_Segmentation(input_im,org)
    seg = mark_seg_in_orgim(input_im,seg)
    return seg

def Select_Roi(med_im,count):
    # Select ROI
    check_image = med_im
    r, c,_ = np.asarray((check_image.shape)) // 2
    roi = check_image[r - r + 10:r + r - 20, c - c + 20:c + c - 20]  # ROI Extraction
    cv2.imwrite('Main/Output/roi/roi_' + str(count) + '.png', roi)
    return roi
def sliding_window(a, window, axis=-1):
    shape = list(a.shape) + [window]
    shape[axis] -= window - 1
    if shape[axis] < 0:
        raise ValueError("Array too small")
    strides = a.strides + (a.strides[axis],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def sliding_img_var(img, window):
    if window <= 0:
        raise ValueError("invalid window size")
    buf = sliding_window(img, 2*window, 0)
    buf = sliding_window(buf, 2*window, 1)

    out = np.zeros(img.shape, dtype=np.float32)
    np.var(buf[:-1,:-1], axis=(-1,-2), out=out[window:-window,window:-window])
    return out

def augment():
    file_path = 'Output/segmented//*.png'
    files = glob.glob(file_path)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    g_path = 'data/gt//*.png'
    files_g = glob.glob(g_path)
    files_g.sort(key=lambda f: int(re.sub('\D', '', f)))
    count = 0
    Feat = []
    label = []
    for i in range(count,len(files)):
        print(files[i])
        input = cv2.imread(files[i])
        input = cv2.resize(input, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        seg = cv2.imread(files_g[i])
        seg = cv2.resize(seg, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        ########## STEP5: Augmentation ###########
        rotate, crop, flip, rand_erasing = Augmentation.Augmentation(input)
        cv2.imwrite(new_aug1_path + "//" + str(count) + '.png', rotate)
        cv2.imwrite(new_aug2_path + "//" + str(count) + '.png', crop)
        cv2.imwrite(new_aug3_path + "//" + str(count) + '.png', flip)
        cv2.imwrite(new_aug4_path + "//" + str(count) + '.png', rand_erasing)
        ############# static_ft################
        ## mean ##
        f1_mean = np.mean(rotate)
        ## variance ##
        gray_rotate = cv2.cvtColor(rotate, cv2.COLOR_BGR2GRAY)
        f1_var = np.var(gray_rotate)
        f1_stat = [f1_mean,f1_var]
        f1_stat = np.array(f1_stat)
        f1_stat = f1_stat.reshape(1,f1_stat.shape[0])
        ## kurthosis ##
        f1_kurt =kurtosis(gray_rotate, axis=0, bias=True)
        f1_kurt = f1_kurt.reshape(1,f1_kurt.shape[0])
        ## Skeweness ##
        f1_skew = skew(gray_rotate, axis=0, bias=True)
        f1_skew = f1_skew.reshape(1,f1_skew.shape[0])

        ## Entropy ##
        entr_img = entropy(gray_rotate, disk(10))
        f1_ent = np.histogram(entr_img[seg], 100)
        f1_ent_fin = f1_ent[0]
        f1_ent_fin = f1_ent_fin.reshape(1,f1_ent_fin.shape[0])
        ################ texture feature ##################
        ## LBP ##
        f1_lbp = lbp.lbp_main(rotate)
        f1_lbp_fin = cv2.calcHist([f1_lbp], [0], None, [100], [0, 256])
        f1_lbp_fin = f1_lbp_fin.reshape(1,f1_lbp_fin.shape[0])
        ## SLBT feat ##
        f1_slbt = SLBT.slbt(rotate)
        f1_slbt_fin = cv2.calcHist([f1_slbt], [0], None, [100], [0, 256])
        f1_slbt_fin = f1_slbt_fin.reshape(1,f1_slbt_fin.shape[0])
        feat1 = np.concatenate((f1_stat, f1_kurt, f1_skew, f1_ent_fin, f1_lbp_fin, f1_slbt_fin), axis=1)
        Feat.append(feat1.tolist())
        ##################################################################################################
        ############# static_ft################
        ## mean ##
        f2_mean = np.mean(crop)
        ## variance ##
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        f2_var = np.var(gray_crop)
        f2_stat = [f2_mean, f2_var]
        f2_stat = np.array(f2_stat)
        f2_stat = f2_stat.reshape(1, f2_stat.shape[0])
        ## kurthosis ##
        f2_kurt = kurtosis(gray_crop, axis=0, bias=True)
        f2_kurt = f2_kurt.reshape(1, f2_kurt.shape[0])
        ## Skeweness ##
        f2_skew = skew(gray_crop, axis=0, bias=True)
        f2_skew = f2_skew.reshape(1, f2_skew.shape[0])

        ## Entropy ##
        entr_img = entropy(gray_crop, disk(10))
        f2_ent = np.histogram(entr_img[seg], 100)
        f2_ent_fin = f2_ent[0]
        f2_ent_fin = f2_ent_fin.reshape(1, f2_ent_fin.shape[0])
        ################ texture feature ##################
        ## LBP ##
        f2_lbp = lbp.lbp_main(crop)
        f2_lbp_fin = cv2.calcHist([f2_lbp], [0], None, [100], [0, 256])
        f2_lbp_fin = f2_lbp_fin.reshape(1, f2_lbp_fin.shape[0])
        ## SLBT feat ##
        f2_slbt = SLBT.slbt(crop)
        f2_slbt_fin = cv2.calcHist([f2_slbt], [0], None, [100], [0, 256])
        f2_slbt_fin = f2_slbt_fin.reshape(1, f2_slbt_fin.shape[0])
        feat2 = np.concatenate((f2_stat, f2_kurt, f2_skew, f2_ent_fin, f2_lbp_fin, f2_slbt_fin), axis=1)
        Feat.append(feat2.tolist())
        ####################################################################################################
        ############# static_ft################
        ## mean ##
        f3_mean = np.mean(flip)
        ## variance ##
        gray_flip = cv2.cvtColor(flip, cv2.COLOR_BGR2GRAY)
        f3_var = np.var(gray_flip)
        f3_stat = [f3_mean, f3_var]
        f3_stat = np.array(f3_stat)
        f3_stat = f3_stat.reshape(1, f3_stat.shape[0])
        ## kurthosis ##
        f3_kurt = kurtosis(gray_flip, axis=0, bias=True)
        f3_kurt = f3_kurt.reshape(1, f3_kurt.shape[0])
        ## Skeweness ##
        f3_skew = skew(gray_flip, axis=0, bias=True)
        f3_skew = f3_skew.reshape(1, f3_skew.shape[0])

        ## Entropy ##
        entr_img = entropy(gray_flip, disk(10))
        f3_ent = np.histogram(entr_img[seg], 100)
        f3_ent_fin = f3_ent[0]
        f3_ent_fin = f3_ent_fin.reshape(1, f3_ent_fin.shape[0])
        ################ texture feature ##################
        ## LBP ##
        f3_lbp = lbp.lbp_main(flip)
        f3_lbp_fin = cv2.calcHist([f3_lbp], [0], None, [100], [0, 256])
        f3_lbp_fin = f3_lbp_fin.reshape(1, f3_lbp_fin.shape[0])
        ## SLBT feat ##
        f3_slbt = SLBT.slbt(flip)
        f3_slbt_fin = cv2.calcHist([f3_slbt], [0], None, [100], [0, 256])
        f3_slbt_fin = f3_slbt_fin.reshape(1, f3_slbt_fin.shape[0])
        feat3 = np.concatenate((f3_stat, f3_kurt, f3_skew, f3_ent_fin, f3_lbp_fin, f3_slbt_fin), axis=1)
        Feat.append(feat3.tolist())
        #########################################################################################################
        ############# static_ft################
        ## mean ##
        f4_mean = np.mean(rand_erasing)
        ## variance ##
        gray_rd_er = cv2.cvtColor(rand_erasing, cv2.COLOR_BGR2GRAY)
        f4_var = np.var(gray_rd_er)
        f4_stat = [f4_mean, f4_var]
        f4_stat = np.array(f4_stat)
        f4_stat = f4_stat.reshape(1, f4_stat.shape[0])
        ## kurthosis ##
        f4_kurt = kurtosis(gray_rd_er, axis=0, bias=True)
        f4_kurt = f4_kurt.reshape(1, f4_kurt.shape[0])
        ## Skeweness ##
        f4_skew = skew(gray_rd_er, axis=0, bias=True)
        f4_skew = f4_skew.reshape(1, f4_skew.shape[0])

        ## Entropy ##
        entr_img = entropy(gray_rd_er, disk(10))
        f4_ent = np.histogram(entr_img[seg], 100)
        f4_ent_fin = f4_ent[0]
        f4_ent_fin = f4_ent_fin.reshape(1, f4_ent_fin.shape[0])
        ################ texture feature ##################
        ## LBP ##
        f4_lbp = lbp.lbp_main(rand_erasing)
        f4_lbp_fin = cv2.calcHist([f4_lbp], [0], None, [100], [0, 256])
        f4_lbp_fin = f4_lbp_fin.reshape(1, f4_lbp_fin.shape[0])
        ## SLBT feat ##
        f4_slbt = SLBT.slbt(rand_erasing)
        f4_slbt_fin = cv2.calcHist([f4_slbt], [0], None, [100], [0, 256])
        f4_slbt_fin = f4_slbt_fin.reshape(1, f4_slbt_fin.shape[0])
        feat4 = np.concatenate((f4_stat, f4_kurt, f4_skew, f4_ent_fin, f4_lbp_fin, f4_slbt_fin), axis=1)
        Feat.append(feat4.tolist())
        #########################################################################################################
        count+=1
    Feat_fin = np.array(Feat)
    Feat_fin = Feat_fin.reshape(Feat_fin.shape[0],Feat_fin.shape[2])
    np.save('Feat_fin.npy',Feat_fin)
    # np.savetxt("Feat.csv",Feat,delimiter=',',fmt='%s')
    # np.savetxt("Label.csv",label,delimiter=',',fmt='%s')
    return Feat

def pre_process():
    # read database and extract features #
    file_path='data/im//*.png'
    gt_path = 'data/gt/*.png'
    files_gt = glob.glob(gt_path)
    files_gt.sort(key=lambda f: int(re.sub('\D', '', f)))
    files = glob.glob(file_path)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    count=0
    for i in range(count,len(files)):
        print(files[i])
        ########## STEP1: Read Database ###########
        input = cv2.imread(files[i])
        org_im = cv2.imread(files_gt[i])
        input_im = cv2.resize(input, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        org_im = cv2.resize(org_im, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        ######### STEP2: ROI Extraction ########
        roi = Select_Roi(input_im,count)  # ROI Extraction
        cv2.imwrite(new_im_path+"//" + str(count) + '.png', roi)
        ######### STEP3: Pre-processing #########
        # t2fcs = T2FCS(roi)
        b = amf(roi[:,:,0], 3, 11)
        g = amf(roi[:,:,1],3,11)
        r = amf(roi[:,:,2],3,11)
        bgr_amf = (np.dstack((b, g, r))).astype(np.uint8)
        # amf_im = amf(roi, 3, 11)
        cv2.imwrite(new_fil_path+"//" + str(count) + '.png', bgr_amf)
        ######### STEP4: SEGNET Segmentation #########
        seg = segment(bgr_amf,org_im)
        cv2.imwrite(new_seg_path + "//" + str(count) + '.png', seg)
        count +=1

def processing():
    pre_process() # ROI,Preprocessing,T2FCS,Segmentation
    augment() # augmentation

def process():
    print("\n >> ROI Extraction..")
    print("\n >> Preprocessing..")
    print("\n >> AMF Filter..")
    print("\n >> Segnet Segmentation..")
    # processing()
