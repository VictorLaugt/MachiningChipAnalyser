import cv2 as cv
import numpy as np


def remove_small_components(img, min_area):
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(img, connectivity=8)

    filtered_img = np.zeros_like(img)
    keep_mask = np.zeros_like(labels, dtype=bool)

    for i in range(1, num_labels):
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            keep_mask |= (labels == i)

    filtered_img[keep_mask] = 255

    return filtered_img


if __name__ == "__main__":
    img = cv.imread("morph2.png", cv.IMREAD_GRAYSCALE)
    filtered_img = remove_small_components(img, min_area=45)

    cv.imshow("original", img)
    cv.imshow("filtered", filtered_img)
    while cv.waitKey(30) != 113:
        pass
    cv.destroyAllWindows()
