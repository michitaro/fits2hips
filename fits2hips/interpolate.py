import numpy


def linear(array, X, Y, dtype=None):
    XI, XR = index_and_ratio(X, dtype)
    YI, YR = index_and_ratio(Y, dtype)
    
    V00 = imageIndexMap(array, XI + 0, YI + 0, dtype)
    V10 = imageIndexMap(array, XI + 1, YI + 0, dtype)
    V01 = imageIndexMap(array, XI + 0, YI + 1, dtype)
    V11 = imageIndexMap(array, XI + 1, YI + 1, dtype)

    return numpy.array(
        (V00 * (1 - XR) + V10 * XR) * (1 - YR) +
        (V01 * (1 - XR) + V11 * XR) * YR
    )


def imageIndexMap(array, X, Y, dtype=None):  # X: int[], Y: int[]
    if dtype is None:
        dtype = array.dtype
    h, w = array.shape
    no_data = numpy.logical_or(
        numpy.logical_or(Y < 0, Y >= array.shape[0]),
        numpy.logical_or(X < 0, X >= array.shape[1]),
    )
    I = Y * w + X
    I %= array.size
    mapped = numpy.array(array.reshape(array.size)[I], dtype)
    mapped[no_data] = float('nan')
    return mapped


def index_and_ratio(x, dtype):
    floor = numpy.floor(x)
    r = numpy.array(x - floor, dtype=dtype)
    return (numpy.array(floor, dtype=int), r)
