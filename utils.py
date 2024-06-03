import numpy as np

def equal_streams(stream1, stream2):
    return all(np.array_equal(img1, img2) for img1, img2 in zip(stream1, stream2))
