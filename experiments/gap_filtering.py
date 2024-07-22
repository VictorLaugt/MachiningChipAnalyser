import numpy as np


def gap_filter(x_values):
    # padding
    x = np.empty(len(x_values)+3, dtype=np.float64)
    x[1:-2] = x_values
    x[0], x[-2], x[-1] = -np.inf, np.inf, np.inf
    print(x)

    # statistics
    gap_q1, gap_median, gap_max = np.percentile(np.diff(x_values), (25, 50, 100))
    gap_ceil = gap_max + gap_q1
    print(f"gap median = {gap_median}")
    print(f"gap ceil     = {gap_ceil}")

    # filtering
    indices = [0]
    left, right = 1, 2
    while right < len(x)-1:
        gap = x[right] - x[left]
        left_gap = x[right] - x[indices[-1]]
        right_gap = x[right+1] - x[left]

        if gap >= gap_median:
            print(f"x[{left}]={x[left]}, x[{right}]={x[right]}, no anomaly")
            indices.append(left)
        elif right_gap < left_gap and right_gap < gap_ceil:
            # remove right point
            print(f"x[{left}]={x[left]}, x[{right}]={x[right]}, remove right point")
            right += 1
            indices.append(left)
        elif left_gap < right_gap and left_gap < gap_ceil:
            # remove left point
            print(f"x[{left}]={x[left]}, x[{right}]={x[right]}, remove left point")
            pass
        else:
            print(f"x[{left}]={x[left]}, x[{right}]={x[right]}, removal would create a too big gap")
            indices.append(left)

        left, right = right, right+1

    return x[indices[1:]]


if __name__ == '__main__':
    # x = np.cumsum([8, 8, 8, 1, 1, 1, 5, 8, 8], dtype=np.float64)
    x = np.cumsum([8, 8, 8, 1, 8, 8], dtype=np.float64)
    print(f"{gap_filter(x)=}")
