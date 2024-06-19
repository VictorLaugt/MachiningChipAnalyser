import numpy as np
import cv2 as cv

import connected_components
import video

from pathlib import Path


log_kernel = 2 * np.array([
    [0, 1, 1, 2, 2, 2, 1, 1, 0],
    [1, 2, 4, 5, 5, 5, 4, 2, 1],
    [1, 4, 5, 3, 0, 3, 5, 4, 1],
    [2, 5, 3, -12, -24, -12, 3, 5, 2],
    [2, 5, 0, -24, -40, -24, 0, 5, 2],
    [2, 5, 3, -12, -24, -12, 3, 5, 2],
    [1, 2, 4, 5, 5, 5, 4, 2, 1],
    [1, 4, 5, 3, 0, 3, 5, 4, 1],
    [0, 1, 1, 2, 2, 2, 1, 1, 0]
])



cam = cv.VideoCapture(0)
# cam = cv.VideoCapture(str(Path("experiments", "input.avi")))



# Real-time image processing loop
processed = []
while True:
    status, img = cam.read()
    if not status:
        break

    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    edge_img = cv.filter2D(gray_img, -1, log_kernel)
    binary_img = cv.threshold(edge_img, 240, 255, cv.THRESH_BINARY)[1]
    blobfilter_img = connected_components.remove_small_components(binary_img, min_area=20)
    morph_img = cv.erode(blobfilter_img, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)), iterations=2)

    processed.append(morph_img)
    if len(processed) == 120:
        break

    # # Check for the 'Enter' key press to exit the loop
    # if cv.waitKey(10) == 13:
    #     break


print("Displaying the processed images...")
video.VideoImgSeqPlayer(processed, "player").play()
cam.release()
cv.destroyAllWindows()



# for img in processed:
#     cv.imshow("processed", img)
#     if cv.waitKey(10) == 120:
#         break

# cam.release()
# cv.destroyAllWindows()
