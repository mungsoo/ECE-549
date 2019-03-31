# def plot_lines_and_vp(im, lines, vp):
    # """
    # Plots user-input lines and the calculated vanishing point.
    # Inputs:
        # im: np.ndarray of shape (height, width, 3)
        # lines: np.ndarray of shape (3, n)
            # where each column denotes the parameters of the line equation
        # vp: np.ndarray of shape (3, )
    # """
    # bx1 = min(1, vp[0] / vp[2]) - 10
    # bx2 = max(im.shape[1], vp[0] / vp[2]) + 10
    # by1 = min(1, vp[1] / vp[2]) - 10
    # by2 = max(im.shape[0], vp[1] / vp[2]) + 10

    # plt.figure()
    # plt.imshow(im)
    # for i in range(lines.shape[1]):
        # if lines[0, i] < lines[1, i]:
            # pt1 = np.cross(np.array([1, 0, -bx1]), lines[:, i])
            # pt2 = np.cross(np.array([1, 0, -bx2]), lines[:, i])
        # else:
            # pt1 = np.cross(np.array([0, 1, -by1]), lines[:, i])
            # pt2 = np.cross(np.array([0, 1, -by2]), lines[:, i])
        # pt1 = pt1 / pt1[2]
        # pt2 = pt2 / pt2[2]
        # plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g')

    # plt.plot(vp[0] / vp[2], vp[1] / vp[2], 'ro')
    # plt.show()
    
# def get_input_lines(im, min_lines=3):
    # """
    # Allows user to input line segments; computes centers and directions.
    # Inputs:
        # im: np.ndarray of shape (height, width, 3)
        # min_lines: minimum number of lines required
    # Returns:
        # n: number of lines from input
        # lines: np.ndarray of shape (3, n)
            # where each column denotes the parameters of the line equation
        # centers: np.ndarray of shape (3, n)
            # where each column denotes the homogeneous coordinates of the centers
    # """
    # n = 0
    # lines = np.zeros((3, 0))
    # centers = np.zeros((3, 0))

    # plt.figure()
    # plt.imshow(im)
    # print('Set at least %d lines to compute vanishing point' % min_lines)
    # while True:
        # print('Click the two endpoints, use the right key to undo, and use the middle key to stop input')
        # clicked = plt.ginput(2, timeout=0, show_clicks=True)
        # if not clicked or len(clicked) < 2:
            # if n < min_lines:
                # print('Need at least %d lines, you have %d now' % (min_lines, n))
                # continue
            # else:
                # # Stop getting lines if number of lines is enough
                # break

        # # Unpack user inputs and save as homogeneous coordinates
        # pt1 = np.array([clicked[0][0], clicked[0][1], 1])
        # pt2 = np.array([clicked[1][0], clicked[1][1], 1])
        # # Get line equation using cross product
        # # Line equation: line[0] * x + line[1] * y + line[2] = 0
        # line = np.cross(pt1, pt2)
        # lines = np.append(lines, line.reshape((3, 1)), axis=1)
        # # Get center coordinate of the line segment
        # center = (pt1 + pt2) / 2
        # centers = np.append(centers, center.reshape((3, 1)), axis=1)

        # # Plot line segment
        # plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='b')

        # n += 1

    # return n, lines, centers
    
    
import matplotlib.pyplot as plt
plt.figure()
plt.show()