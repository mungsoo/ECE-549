import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import correlate as conv
from scipy.signal import medfilt2d
IMG_DIR = 'images/'
MOSAIC_IMG_NAME = 'tony.bmp'
ORI_IMG_NAME = 'tony.jpg'
def read_image(IMG_NAME):
    # YOUR CODE HERE
    img = cv2.imread(IMG_NAME)
    return img

# For a sanity check, display your image here
# mosaic_img = read_image(IMG_DIR + MOSAIC_IMG_NAME)[:,:,0]# YOUR CODE HERE
# cv2.imshow('image', mosaic_img)
# cv2.waitKey(0)

def get_solution_image(mosaic_img):
    '''
    This function should return the soln image.
    Feel free to write helper functions in the above cells
    as well as change the parameters of this function.
    '''
    mosaic_shape = np.shape(mosaic_img)
    # soln_image = np.zeros((mosaic_shape[0], mosaic_shape[1], 3))
    ### YOUR CODE HERE ###

    # Make sure broadcast works correctly
    assert mosaic_shape[0] % 2 == 0 and mosaic_shape[1] % 2 == 0
    print(mosaic_shape)
    width_n = mosaic_shape[1] // 2
    height_n = mosaic_shape[0] // 2
    # Extract channels
    red_channel = mosaic_img * np.tile(np.array([[1, 0], [0, 0]]), (height_n, width_n))
    green_channel = mosaic_img * np.tile(np.array([[0, 1], [1, 0]]), (height_n, width_n))
    blue_channel = mosaic_img * np.tile(np.array([[0, 0], [0, 1]]), (height_n, width_n))

    # Do conv for each channel
    red_channel = conv(red_channel, np.array([[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]]),
                       output=np.dtype('uint8'), mode='mirror')
    green_channel = conv(green_channel, np.array([[0, 0.25, 0], [0.25, 1, 0.25], [0, 0.25, 0]]),
                       output=np.dtype('uint8'), mode='mirror')
    blue_channel = conv(blue_channel, np.array([[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]]),
                       output=np.dtype('uint8'), mode='mirror')

    soln_image = np.array([blue_channel, green_channel, red_channel]).transpose((1, 2, 0))
    return soln_image


def compute_errors(soln_image, original_image):
    '''
    Compute the Average and Maximum per-pixel error
    for the image.

    Also generate the map of pixel differences
    to visualize where the mistakes are made
    '''
    err_map = ((soln_image.astype('float32') - original_image.astype('float32')) ** 2).sum(2)
    print(err_map)
    plt.imshow(err_map, 'gray')
    plt.show()
    size = np.shape(err_map)[0] * np.shape(err_map)[1]
    pp_err = err_map.sum() / size
    max_err = err_map.max()
    return pp_err, max_err

# mosaic_img = get_solution_image(mosaic_img)
# print(mosaic_img)
# cv2.imshow('image', mosaic_img)
# cv2.waitKey(0)
# origin_img = read_image(IMG_DIR + ORI_IMG_NAME)
# (pp_err, max_err) = compute_errors(mosaic_img, origin_img)
# print(pp_err, max_err)


# def get_freeman_solution_image(mosaic_img):
    # '''
    # This function should return the freeman soln image.
    # Feel free to write helper functions in the above cells
    # as well as change the parameters of this function.

    # HINT : Use the above get_solution_image function.
    # '''
    # ### YOUR CODE HERE ###
    # mosaic_shape = np.shape(mosaic_img)
    # # soln_image = np.zeros((mosaic_shape[0], mosaic_shape[1], 3))
    # ### YOUR CODE HERE ###

    # # Make sure broadcast works correctly
    # assert mosaic_shape[0] % 2 == 0 and mosaic_shape[1] % 2 == 0
    # width_n = mosaic_shape[1] // 2
    # height_n = mosaic_shape[0] // 2

    # red_mask = np.tile(np.array([[1, 0], [0, 0]]), (height_n, width_n))
    # green_mask = np.tile(np.array([[0, 1], [1, 0]]), (height_n, width_n))
    # blue_mask = np.tile(np.array([[0, 0], [0, 1]]), (height_n, width_n))

    # # Extract channels
    # red_channel = mosaic_img * red_mask
    # green_channel = mosaic_img * green_mask
    # blue_channel = mosaic_img * blue_mask

    # # Do conv for each channel
    # green_out = conv(green_channel, np.array([[0, 0.25, 0], [0.25, 1, 0.25], [0, 0.25, 0]]),
                         # output=np.dtype('float32'), mode='mirror')
    # red_out = medfilt2d(conv(red_channel, np.array([[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]]),
                       # output=np.dtype('float32'), mode='mirror') - green_out)
    # blue_out = medfilt2d(conv(blue_channel, np.array([[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]]),
                        # output=np.dtype('float32'), mode='mirror') - green_out)
    # red_out = (red_out + green_out) * (1 - red_mask) + red_channel
    # blue_out = (blue_out + green_out) * (1 - blue_mask) + blue_channel

    # freeman_soln_image = np.array([blue_out, green_out, red_out]).transpose((1, 2, 0))
    # freeman_soln_image[freeman_soln_image<0], freeman_soln_image[freeman_soln_image>255] = 0, 255
    # return freeman_soln_image.astype(np.dtype('uint8'))
def get_freeman_solution_image(mosaic_img):
    '''
    This function should return the freeman soln image.
    Feel free to write helper functions in the above cells
    as well as change the parameters of this function.

    HINT : Use the above get_solution_image function.
    '''
    ### YOUR CODE HERE ###
    freeman_soln_image = \
    get_solution_image(mosaic_img).transpose((2, 0, 1)).astype(np.dtype('float32'))
    freeman_soln_image[0] = \
    medfilt2d(freeman_soln_image[0] - freeman_soln_image[1]) + freeman_soln_image[1]
    freeman_soln_image[2] = \
    medfilt2d(freeman_soln_image[2] - freeman_soln_image[1]) + freeman_soln_image[1]
    freeman_soln_image = \
    freeman_soln_image.transpose((1, 2, 0))
    freeman_soln_image[freeman_soln_image<0], \
    freeman_soln_image[freeman_soln_image>255] = 0, 255
    return freeman_soln_image.astype(np.dtype('uint8'))

def get_mosaic_image(original_image):
    '''
    Generate the mosaic image using the Bayer Pattern.
    '''
    origin_shape = np.shape(original_image)

    # Make sure broadcast works correctly
    assert origin_shape[0] % 2 == 0 and origin_shape[1] % 2 == 0

    width_n = origin_shape[1] // 2
    height_n = origin_shape[0] // 2
    original_image = original_image.transpose((2, 0, 1))

    red_mask = np.tile(np.array([[1, 0], [0, 0]]), (height_n, width_n))
    green_mask = np.tile(np.array([[0, 1], [1, 0]]), (height_n, width_n))
    blue_mask = np.tile(np.array([[0, 0], [0, 1]]), (height_n, width_n))

    red_c = original_image[2] * red_mask
    green_c = original_image[1] * green_mask
    blue_c = original_image[0] * blue_mask

    mosaic_img = red_c + green_c + blue_c

    return mosaic_img

# mosaic_img = read_image(IMG_DIR + 'crayons.bmp')[:,:,0]
# mosaic_img = get_freeman_solution_image(mosaic_img)
# cv2.imwrite(IMG_DIR + 'crayons_1.jpg', mosaic_img)
origin_img = read_image(IMG_DIR + 'seasons.jpeg')
mosaic_img = get_mosaic_image(origin_img)
mosaic_img = get_freeman_solution_image(mosaic_img)
cv2.imwrite(IMG_DIR + 'seasons_1.jpg', mosaic_img)
(pp_err, max_err) = compute_errors(mosaic_img, origin_img)
print(pp_err, max_err)