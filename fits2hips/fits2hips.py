import argparse
import os
import hashlib
import math
import logging
import sys
import glob
import multiprocessing
import itertools
import six
import shutil
logging.basicConfig(level=logging.INFO)

# from PIL import Image
import numpy
import astropy.io.fits as afits
import astropy.wcs as awcs
import healpy
from . import parallel
from . import hips
from . import interpolate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', '-o', required=True)
    parser.add_argument('--work-dir', '-w', default='work')
    parser.add_argument('src', nargs='+')
    parser.add_argument('--filter', '-f', required=True, choices='g r i z y'.split())
    parser.add_argument('--tile-size-order', type=int, default=8)
    parser.add_argument('--pixel-scale', type=lambda s: numpy.deg2rad(float(s) / 3600.))
    parser.add_argument('--max-order', type=int)
    parser.add_argument('--debug', '-D', action='store_true')
    args = parser.parse_args()

    args.src = filter(use(args), args.src)

    if args.pixel_scale is None:
        args.pixel_scale = guess_pixel_scale(args.src[0])
    pixel_order = best_order_for(pixel_scale=args.pixel_scale)
    tile_order = pixel_order - args.tile_size_order
    logging.info('tile order: {}'.format(tile_order))

    warp_source_files(args.src, args.work_dir, pixel_order, tile_order)
    stack_tiles(args.work_dir, pixel_order, tile_order)
    # build_table(args.work_dir, tile_order)
    make_lower_order_tiles(args.out_dir, args.work_dir,
                           tile_order, pixel_order)


def use(args):
    def f(fname):
        return hdul[0].header['FPA.FILTER'] == '{}.00000'.format(args.filter)
    return f


def make_lower_order_tiles(out_dir, work_dir, tile_order, pixel_order):
    tile_size = 1 << (pixel_order - tile_order)
    sub_size = tile_size // 2

    tile_indices = set()
    for j in glob.iglob('{work_dir}/{tile_order}/*/*/stack.fits'.format(**locals())):
        tile_index = int(j.split('/')[-2])
        tile_indices.add(tile_index >> 2)
        dir_name = (tile_index // 10000) * 10000
        out_file = '{out_dir}/Norder{tile_order}/Dir{dir_name}/Npix{tile_index}.fits'.format(
            **locals())
        mkdir_p(os.path.dirname(out_file))
        shutil.copyfile(j, out_file)

    for target_order in range(tile_order - 1, -1, -1):

        def src_fname(index):
            # import parent scope
            work_dir
            out_dir
            # --
            src_order = target_order + 1
            dir_name = (index // 10000) * 10000
            return '{out_dir}/Norder{src_order}/Dir{dir_name}/Npix{index}.fits'.format(**locals())

        def gather_children(target_index):
            logging.info('tile: {}/{}'.format(target_order, target_index))

            # import parent scope
            out_dir
            target_order
            # --
            base_index = target_index << 2
            # sub_tile = [(0, 0), (0, 1), (1, 0), (1, 1)]
            sub_tile = [(0, 1), (0, 0), (1, 1), (1, 0)]
            buf = numpy.empty((tile_size, tile_size), dtype=numpy.float32)
            buf.fill(float('nan'))
            for sub_index, (j, i) in enumerate(sub_tile):
                fname = src_fname(base_index + sub_index)
                if os.path.exists(fname):
                    with afits.open(fname) as hdul:
                        hdu = find_image_hdu(hdul)
                        sub_data = half(hdu.data)
                        dd = sub_size
                        ii = dd * i
                        jj = dd * j
                        buf[ii: ii + dd, jj: jj + dd] = sub_data

            dir_name = (target_index // 10000) * 10000
            out_file = '{out_dir}/Norder{target_order}/Dir{dir_name}/Npix{target_index}.fits'.format(
                **locals())
            mkdir_p(os.path.dirname(out_file))
            afits.HDUList([afits.PrimaryHDU(buf)]).writeto(
                out_file, output_verify='fix', overwrite=True)

        parallel.map(gather_children, list(tile_indices))
        new_indices = set()
        for i in tile_indices:
            new_indices.add(i >> 2)
        tile_indices = new_indices


def flip_v(array):
    return array[::-1]


def half(array):
    h, w = array.shape
    # return numpy.nanmean(numpy.nanmean(array.reshape((h // 2, 2, w // 2, 2)), axis=3), axis=1)
    return numpy.nanmedian(numpy.nanmedian(array.reshape((h // 2, 2, w // 2, 2)), axis=3), axis=1)


def interpolate_nan(line):
    nans = numpy.isnan(line)
    if numpy.any(nans):
        nan_indices = numpy.nonzero(nans)[0]
        val_indices = ~nan_indices
        line[nans] = numpy.interp(nan_indices, val_indices, line[val_indices])


def repair_image(a):
    for i in range(len(a)):
        interpolate_nan(a[i])


# def build_table(work_dir, tile_order):
#     ntiles = 12 * (1 << (2 * tile_order))
#     nbytes = ntiles // 8
#     logging.info('building table: {} MB'.format(nbytes // 1000000))
#     table = numpy.zeros(nbytes, dtype='uint8')
#     for tile_file in glob.iglob('{work_dir}/{tile_order}/*/*/stack.fits'.format(**locals())):
#         tile_index = int(tile_file.split('/')[-2])
#         byte_index = tile_index >> 3  # == tile_index // 8
#         bit_index = tile_index & 7   # == tile_index %  8
#         table[byte_index] |= (128 >> bit_index)
#     return table

# def bit_shift_table(table, s):
#     if s == 0:
#         return
#     assert abs(s) <= 7


def warp_source_files(fnames, work_dir, pixel_order, tile_order):
    tile_nside = healpy.order2nside(tile_order)
    for fname in fnames:
        with afits.open(fname) as hdul:
            src = Source(find_image_hdu(hdul))
            tile_indices = healpy.query_polygon(
                tile_nside, ad2xyz(*src.polygon).T, nest=True)
            logging.info('{} tiles'.format(len(tile_indices)))

            def warp(tile_index):
                logging.info('warping {}:{}...'.format(fname, tile_index))
                src.warp(work_dir, tile_order, tile_index, pixel_order)

            parallel.map(warp, tile_indices)


def find_image_hdu(hdul):
    for hdu in hdul:
        if hdu.data is not None and hdu.header['NAXIS'] == 2:
            return hdu


def stack_tiles(work_dir, pixel_order, tile_order):
    batch_size = 100 * multiprocessing.cpu_count()
    for tile_indices in batch(glob.iglob('{work_dir}/{tile_order}/*/*'.format(**locals())), batch_size):
        def stack(tile_index):
            out_file = '{tile_index}/stack.fits'.format(**locals())
            warps = glob.glob('{tile_index}/warp-*.fits'.format(**locals()))
            logging.info(
                'stacking {}/{}: {} tiles'.format(tile_order, tile_index, len(warps)))
            stacked = six.moves.reduce(
                merge_data, six.moves.map(tile_data,  warps))
            afits.HDUList([afits.PrimaryHDU(stacked)]).writeto(
                out_file, output_verify='fix', overwrite=True)
        parallel.map(stack, list(tile_indices))


def merge_data(a, b):
    ok = numpy.isfinite(b)
    a[ok] = b[ok]
    return a


def tile_data(fname):
    with afits.open(fname, memmap=False) as hdul:
        hdu = find_image_hdu(hdul)
        return hdu.data


def batch(iterable, size):
    # https://stackoverflow.com/a/8290514/2741327
    from itertools import islice, chain
    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, size)

        yield chain([six.next(batchiter)], batchiter)


def best_order_for(pixel_scale):
    d = pixel_scale
    best_o = 1. / 2. * math.log(math.pi / (3. * d * d)) / math.log(2)
    o = int(best_o + 1)
    logging.info('pixel scale (arcsec): {:.4f} -> {:.4f}'.format(rad2arcsec(pixel_scale),
                                                                 rad2arcsec(healpy.nside2resol(healpy.order2nside(o)))))
    return o


def rad2arcsec(rad):
    return numpy.rad2deg(rad) * 3600.


class Source(object):
    def __init__(self, hdu):
        self.hdu = hdu
        self.checksum = self._checksum(hdu)
        self.wcs = awcs.WCS(self.hdu.header)
        self.polygon = self._polygon()

    def warp(self, out_dir, tile_order, tile_index, pixel_order):
        dir_name = (tile_index // 10000) * 10000
        checksum = self.checksum
        out_file = '{out_dir}/{tile_order}/{dir_name}/{tile_index}/warp-{checksum}.fits'.format(
            **locals())
        if not os.path.exists(out_file):
            mkdir_p(os.path.dirname(out_file))
            mapped = self.mapped_tile(out_file,  tile_order,
                                      tile_index, pixel_order)
            # repair_image(mapped)
            mapped = flip_v(mapped)
            afits.HDUList([afits.PrimaryHDU(mapped)]).writeto(
                out_file, output_verify='fix')

    def _checksum(self, hdu):
        m = hashlib.md5()
        l = hdu.data.size
        m.update(hdu.data[l // 2: l // 2 + 5 * l // 100])
        m.update(str(hdu.header).encode('utf-8'))
        return m.hexdigest()

    def _polygon(self):
        m = 256
        h = self.hdu.header
        X, Y = numpy.array([
            [-m, -m],
            [h['NAXIS1'] + m, -m],
            [h['NAXIS1'] + m, h['NAXIS2'] + m],
            [-m, h['NAXIS2'] + m],
        ]).T
        A, D = self.wcs.wcs_pix2world(X, Y, 0)
        logging.info('corners: {}'.format(repr(numpy.array([A, D]).T)))
        return numpy.deg2rad((A, D))

    def mapped_tile(self, out_file, tile_order, tile_index, pixel_order):
        tile_size_order = pixel_order - tile_order
        tile_nside = healpy.order2nside(tile_size_order)
        pixels_per_tile = tile_nside * tile_nside
        pixel_nside = healpy.order2nside(pixel_order)
        pixel_index_start = tile_index << (2 * tile_size_order)
        pixel_indices = numpy.arange(
            pixel_index_start, pixel_index_start + pixels_per_tile)[hips.nest2ring(tile_nside)]
        THETA, PHI = healpy.pix2ang(pixel_nside, pixel_indices, nest=True)
        RA = numpy.rad2deg(PHI)
        DEC = numpy.rad2deg(numpy.pi / 2 - THETA)
        X, Y = self.wcs.wcs_world2pix(RA, DEC, 0)
        mapped = interpolate.linear(self.hdu.data, X, Y, dtype=numpy.float32).reshape(
            (tile_nside, tile_nside))
        return mapped


def guess_pixel_scale(fname):
    # TODO: calculate pixel scale from wcs
    # according to CD matrix or CDELTA
    # return numpy.deg2rad(0.168 / 3600.)  # for HSC
    return numpy.deg2rad(0.25 / 3600.)  # for PS1


def ad2xyz(a, d):
    cosd = numpy.cos(d)
    return numpy.array((
        cosd * numpy.cos(a),
        cosd * numpy.sin(a),
        numpy.sin(d)
    ))


def mkdir_p(dirName):
    import os
    try:
        os.makedirs(dirName)
    except OSError:
        pass


if __name__ == '__main__':
    main()
