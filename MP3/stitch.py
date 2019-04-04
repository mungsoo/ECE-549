import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
from match import match
from skimage.transform import warp

class stitch:
    
    def __init__(self, args):
        self.path = args
        self.matcher = match()
        self.images = None
        
        if args:
            self.read_images(args)
            
    def read_images(self, args):
        self.path = args
        fp = open(self.path, 'r')
        filenames = [file.rstrip("\r\n") for file in fp.readlines()]
        print("Stitching :", filenames)
        self.images = [plt.imread(file) for file in filenames]
        
    def stitch2(self, left_img, right_img, verbose=False):
        
        
        def get_warped(left_img, right_img, H, H_affine, dsize):
    
            H = H_affine @ H
            H_inv = np.linalg.inv(H)
            
            warped = np.zeros(np.append(3, dsize), dtype=np.float64)
            for i in range(3):
                warped[i] = warp(left_img[:, :, i].astype("float64"), H_inv, output_shape=dsize)
                warped_right = warp(right_img[:, :, i].astype("float64"), np.linalg.inv(H_affine), output_shape=dsize)
                warped_right[warped[i] != 0] = 0
                warped[i] += warped_right
            warped = warped.transpose(1, 2, 0)
            return warped
        
        def get_H_affine_dsize(H, left_shape, right_shape):
            """
            @H : transformation matrix from left to right
            """
            H_affine = np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            left_corner_indices = (H @ np.array([[0, 0, 1], [left_shape[1], left_shape[0], 1],\
                                       [0, left_shape[0], 1], [left_shape[1], 0, 1]]).T).T
            left_corner_indices /= left_corner_indices[:, -1].reshape(-1, 1)
        
            offset_x = min(np.append(left_corner_indices[:, 0], 0))
            offset_y = min(np.append(left_corner_indices[:, 1], 0))
        
            H_affine[0][-1] -= offset_x
            H_affine[1][-1] -= offset_y
        
            left_corner_indices = (H_affine @ left_corner_indices.T).T
            right_corner_indices = (H_affine @ np.array([[0, 0, 1], [right_shape[1], right_shape[0], 1],\
                                       [0, right_shape[0], 1], [right_shape[1], 0, 1]]).T).T
        
            dsize = np.zeros((2), dtype=np.int64)
            dsize[1] = max(max(left_corner_indices[:, 0]), max(right_corner_indices[:, 0]))
            dsize[0] = max(max(left_corner_indices[:, 1]), max(right_corner_indices[:, 1]))
            
            return H_affine, dsize

        gray_left= cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right= cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        H, inlier_idxs = self.matcher.get_match(gray_left, gray_right, verbose=verbose)
        H_affine, dsize = get_H_affine_dsize(H, gray_left.shape, gray_right.shape)
        
        warped = get_warped(left_img, right_img, H, H_affine, dsize)
        
        return warped
    
    def stitchn(self, verbose=False):
        
        warped = self.images[0]
        for i in range(1, len(self.images)):
            unwarped = self.images[i]
            warped = self.stitch2(warped.astype("uint8"), unwarped, verbose=verbose)
        
        return warped
        
        
        
if __name__ == "__main__":
    
    try:
        args = sys.argv[1]
    except:
        args = "files/file4.txt"
    finally:
        print("Parameters : ", args)
    
    stitch_obj = stitch(args)
    warped = stitch_obj.stitchn()
    plt.figure()
    plt.imshow(warped.astype("uint8"))
    plt.show()