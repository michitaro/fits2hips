import numpy


_ring2nestCache = {}


def ring2nest(nside):
    l = nside
    if l not in _ring2nestCache:
        idx = numpy.arange(l * l)
        X = shrinkBit(idx)
        Y = shrinkBit(idx << 1)
        _ring2nestCache[l] = l * Y + X
    index = _ring2nestCache[l]
    return index


_nest2ringCache = {}


def nest2ring(nside):
    l = nside
    if l not in _nest2ringCache:
        X, Y = numpy.meshgrid(numpy.arange(l), numpy.arange(l))
        _nest2ringCache[l] = (sandwichZero(X) | (
            sandwichZero(Y) >> 1)).flatten()
    index = _nest2ringCache[l]
    return index


def sandwichZero(n, bits=31):
    m = numpy.zeros_like(n)
    o = numpy.ones_like(n)
    for i in range(bits):
        m |= (n & (o << i)) << (i + 1)
    return m


def shrinkBit(n, bits=31):
    m = numpy.zeros_like(n)
    for i in range(bits):
        m |= (n & (2 << (2 * i))) >> (i + 1)
    return m