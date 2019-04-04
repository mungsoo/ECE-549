import cv2
import numpy as np
import random
from scipy.spatial.distance import cdist

class match:
    def get_match(self, gray_left, gray_right, verbose=False):
        
        def get_avg_residual(model, matches):
            def augment(data):
                aug_data = np.ones((len(data), 3), dtype=np.float64)
                aug_data[:,:2] = data
                return aug_data
            x, x_p = augment(matches[:,0]), augment(matches[:,1])
            # print(model.params)
            # print(x_p)

            result = model @ x.T
            result /= result[-1]
            result = result.transpose(1, 0)
            return np.mean(result - x_p)
        
        sift = cv2.xfeatures2d.SIFT_create()
        
        left_keypoints, left_neighborhood_list = sift.detectAndCompute(gray_left,None)
        right_keypoints, right_neighborhood_list = sift.detectAndCompute(gray_right,None)
        dist = cdist(left_neighborhood_list, right_neighborhood_list, 'sqeuclidean')
        
        putative_match = self.stable_match(dist, t=16000)
        left_keypoints = np.array([left_keypoints[idx].pt for idx in putative_match[0]])
        right_keypoints = np.array([right_keypoints[idx].pt for idx in putative_match[1]])
        
        indices = np.stack((left_keypoints, right_keypoints), axis=1).astype("uint64")
        
        
        H, inlier_idxs = self.ransac_homography(indices[:, [1,0]], threshold=0.5)
        # H, s = cv2.findHomography(indices[:,0], indices[:,1], cv2.RANSAC, 4)
        print("Average inliers residual :" , get_avg_residual(H, indices[inlier_idxs]))
        print("Num of inliers :", len(inlier_idxs))
        
        if verbose:
            import matplotlib.pyplot as plt
            from skimage.feature import plot_matches
            fig, ax =plt.subplots(nrows=1, ncols=1)
            plot_matches(ax, gray_left, gray_right, indices[:, 0, [1, 0]], indices[:, 1, [1, 0]],\
                         np.column_stack((np.arange(indices.shape[0]), np.arange(indices.shape[0]))), matches_color='r')
            plt.title("Putative Matches")        
            fig, ax =plt.subplots(nrows=1, ncols=1)
            plot_matches(ax, gray_left, gray_right, indices[:, 0, [1, 0]], indices[:, 1, [1, 0]],
                         np.column_stack((inlier_idxs, inlier_idxs)), matches_color='b')
            plt.title("Inliers")
        
        return H, inlier_idxs
    
    def stable_match(self, dist, t=2000):
        """
        @dist: ndarray, distance matrix
        @   t: float, threshold used to eliminate matches
        return N x 2 ndarray of matching indices
        """
        dist_shape = dist.shape
        if dist_shape[0] > dist_shape[1]:
            dist = dist.transpose(1, 0)
        row_order = np.empty_like(dist, dtype=np.uint64)
        col_order = np.empty_like(dist, dtype=np.uint64)
        index_list = np.zeros(len(dist)).astype("uint64")
    
        match = np.full(len(dist[0]), -1)
        unmatch = list(np.arange(len(dist)-1, -1, -1))
    
        for i in range(len(dist)):
            row_order[i] = np.argsort(dist[i])
        for i in range(len(dist[0])):
            order = np.argsort(dist[:,i])
            for j in range(len(dist)):
                col_order[order[j]][i] = j
    
        while unmatch:
            row = unmatch.pop(-1)
            while True:
                cur_col = row_order[row, index_list[row]]
                prev_match = match[cur_col]
                if prev_match == -1:
                    index_list[row] += 1
                    match[cur_col] = row
                    break
                elif col_order[prev_match, cur_col] > col_order[row, cur_col]:
                    index_list[row] += 1
                    match[cur_col] = row
                    unmatch.append(prev_match)
                    break
                index_list[row] += 1
    
        
        putative_match = [[], []]
        for i, v in enumerate(match):
            if v != -1 and dist[v][i] < t:
                putative_match[0].append(v)
                putative_match[1].append(i)
        putative_match = np.array(putative_match)
        if dist_shape[0] > dist_shape[1]:
            putative_match[[0, 1], :] = putative_match[[1, 0], :]
        print("Number of putative matches: ", putative_match.shape[1])
        return putative_match
    
    
    
    def ransac_homography(self, indices, threshold, max_iterations=3000000):
        '''
        @indices: a N x 2 x 2 matrix, where each line is x' and x. We want to \
        find the least square solution of lambda x' = Hx
        @threshold: the maximum distance of inliers
        '''
        
        def augment(data):
            aug_data = np.ones((len(data), 3), dtype=np.float64)
            aug_data[:,:2] = data
            return aug_data
    
        def is_inlier(model, x, x_p, threshold=threshold):
            tf_x = model @ x
            tf_x /= tf_x[-1]
            return np.sum((tf_x - x_p)**2) <= threshold**2
            
        num_points = len(indices)
        max_num_inliers = 0
        x_p, x = augment(indices[:,0]), augment(indices[:,1])
        # Construct A
        A = np.zeros((2*len(x), 9), dtype=np.float64)
        for i in range(len(x)):
            A[2*i][3:] = np.hstack((x[i], -x_p[i][1]*x[i]))
            A[2*i+1][:3] = x[i]
            A[2*i+1][6:] = -x_p[i][0]*x[i]
        # print(A)
        
        # Adaptive RANSAC
        N = float('inf')
        sample_count = 0
        N= 1
        optim_model = None
        optim_inliers_idx = []
        while N > sample_count and sample_count < max_iterations:
            rand_index = random.sample(range(num_points), 4)
            
            # Solve for Ah = 0
            # Construct A_sample
            A_sample = np.zeros((8, 9), dtype=np.float64)
            for i in range(4):
                A_sample[2*i:2*i+2] = A[2*rand_index[i]:2*rand_index[i]+2]
            
            # Get the eigenvector lf A^TA corresponding to the smallest eigenvalue
            # This is equivalent to get the vector in V (SVD of A) corresponding to the smallest singular value
            model = np.linalg.svd(A_sample)[-1][-1,:].reshape((3, 3))
            
            # normalize the last element of the model to 1, it seems unnecessary? 
            model /= model[-1][-1]
            # print(model)
            
            num_inliers = 0
            for i in range(num_points):
                if is_inlier(model, x[i], x_p[i]):
                    num_inliers += 1
            
            if num_inliers > max_num_inliers:
                max_num_inliers = num_inliers
                optim_model = model
                N = np.log(0.01) / np.log(1 - np.power(num_inliers / num_points, 4))
                
            sample_count += 1
        
        
        for i in range(num_points):
            if is_inlier(optim_model, x[i], x_p[i]):
                    optim_inliers_idx.append(i)
                    
        # Refit all inliers
        # A_sample = np.zeros((2 * len(optim_inliers_idx), 9), dtype=np.float64)
        # for i in range(len(optim_inliers_idx)):
            # A_sample[2*i:2*i+2] = A[2*optim_inliers_idx[i]:2*optim_inliers_idx[i]+2]
        # optim_model = np.linalg.svd(A_sample)[-1][-1,:].reshape((3, 3))
            
        return optim_model, np.array(optim_inliers_idx)
    
   
