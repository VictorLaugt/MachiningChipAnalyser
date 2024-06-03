import cv2 as cv
import numpy as np


def remove_small_components(img, min_area=45):
    analysis = cv.connectedComponentsWithStats(img, 8, cv.CV_32S)
    total_labels, label_ids, values, centroids = analysis

    filtered_img = np.zeros_like(img)

    for i in range(1, total_labels):
        if values[i, cv.CC_STAT_AREA] >= min_area:
            component = np.where(label_ids == i, 255, 0).astype(np.uint8)
            filtered_img |= component

    return filtered_img


if __name__ == "__main__":
    img = cv.imread("morph2.png", cv.IMREAD_GRAYSCALE)
    filtered = remove_small_components(img, min_area=45)
    cv.imshow("original", img)
    cv.imshow("filtered", filtered)
    while cv.waitKey(30) != 113:
        pass
    cv.destroyAllWindows()
