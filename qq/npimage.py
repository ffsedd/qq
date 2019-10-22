#!/usr/bin/env python3
import imghdr
import numpy as np
import logging
from pathlib import Path
from send2trash import send2trash
from tkinter import filedialog
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from skimage import img_as_float, img_as_ubyte, img_as_uint
from imageio import imread, imwrite
from qq.nptools import normalize, info

FILETYPES = ['jpeg', 'bmp', 'png', 'tiff']


class npImage():
    ''' load image in float numpy array, read metadata '''

    def __init__(self, fpath=None, img_arr=None, fft=None):
        self.fpath = fpath
        self.arr = img_arr
        self.filetype = None
        self.filesize = 0
        self.color_model = 'gray'
        if fpath:
            self.load(fpath)
        logging.debug(f"npImage: shape: {self.arr.shape} | bitdepth: \
                      {self.bitdepth} | color_model: {self.color_model}")

    def properties(self):
        prop = f"{self.fpath}    |  \
                {self.bitdepth}bit {self.filetype}  |  \
                {self.filesize/2**20:.2f} MB  |  \
                {self.width} x {self.height} x {self.channels}  |  \
                color:{self.color_model}"
        return prop

    def __repr__(self):
        return self.properties()

    def _check_filetype(self):
        filetype = imghdr.what(self.fpath)
        assert filetype in FILETYPES, f"Error, not supported filetype: \
        {filetype} - {self.fpath}"
        return filetype

    @property
    def channels(self):
        return 1 if self.arr.ndim == 2 else self.arr.shape[2]

    def color_model_change(self, model):
        if model == self.color_model:  # do not change anything
            return
        elif model == 'rgb' and self.color_model == 'hsv':  # HSV -> RGB
            self.arr = hsv_to_rgb(self.arr)
        elif model == 'hsv' and self.color_model == 'rgb':  # RGB -> HSV
            self.arr = rgb_to_hsv(self.arr)
        elif model == 'gray' and self.color_model == 'rgb':  # RGB -> GRAY
            self.arr = np.dot(self.arr[..., :3], [0.2989, 0.5870, 0.1140])
        elif model == 'rgb' and self.color_model == 'gray':  # GRAY -> RGB
            self.arr = np.stack((self.arr)*3, axis=-1)
        elif model == 'hsv' and self.color_model == 'gray':  # GRAY -> HSV
            self.arr = np.stack((np.zeros_like(self.arr))*2, self.arr, axis=-1)
        else:
            raise f"Conversion not supported {self.color_model} -> {model}"

        self.color_model = model  # conversion done, update mode

    def load(self, fpath=None):
        if not fpath:
            logging.debug("fpath input dialog")
            fpath = filedialog.askopenfilename()
        if not fpath:
            return

        logging.debug(f"open file {fpath}")

        Fpath = Path(fpath)
        self.name = Fpath.stem
        self.filesize = Fpath.stat().st_size
        self.fpath = Fpath
        self.filetype = self._check_filetype()

        self.arr = imread(Fpath)
        self.color_model = 'gray' if self.channels == 1 else 'rgb'
        # get orig bitdepth before conversion to float
        self.bitdepth = self._get_bitdepth(self.arr)
        self.arr = img_as_float(self.arr)  # convert to float

    @property
    def center(self):
        x, y = (size//2 for size in self.arr.shape[:2])
        return x, y

    @property
    def width(self):
        return self.arr.shape[1]

    @property
    def height(self):
        return self.arr.shape[0]

    @property
    def ratio(self):
        return self.arr.shape[0] / self.arr.shape[1]

    def save(self, fpath=None, bitdepth=None):

        fpath = fpath or self.fpath
        bitdepth = bitdepth or self.bitdepth

        Fp = Path(fpath)
        logging.debug(f"save to {Fp} bitdepth:{self.bitdepth} \
                      filetype:{self.filetype}")

        if Fp.is_file():
            send2trash(str(Fp))

        self._save_image(self.arr, fpath=fpath, bitdepth=bitdepth)
        self.fpath = fpath

    def rotate(self, angle=90):
        ''' rotate array by 90 degrees or more '''
        self.arr = im_rotate(self.arr, angle)

    def normalize(self):
        '''   '''
        self.arr = normalize(self.arr)

    def to_gray(self):
        '''   '''
        self.color_model_change("gray")

    def gamma(self, g):
        self.arr = self.arr ** g

    def _save_image(self, float_arr, fpath, bitdepth=8):
        ''' '''

        assert isinstance(float_arr, (np.ndarray, np.generic))

        Fp = Path(fpath)
        Fp.parent.mkdir(exist_ok=True)

        float_arr = np.clip(float_arr, a_min=0, a_max=1)

        arr = self._float_to_int(float_arr, bitdepth)

        imwrite(Fp, arr)

        logging.debug(f"image saved")

    def _float_to_int(self, arr, bitdepth=8):
        ''' '''
        if bitdepth == 8:
            return img_as_ubyte(arr)
        else:
            return img_as_uint(arr)

    def free_rotate(self, angle):
        ''' rotate array
        '''
        from scipy.ndimage import rotate
        self.arr = rotate(self.arr, angle,
                          reshape=True, mode='nearest')

        self.info()
        self.arr = np.clip(self.arr, 0, 1)

    def crop(self, x0, y0, x1, y1):

        logging.debug(f"apply crop: {x0} {x1} {y0} {y1}")
        self.arr = self.arr[self.slice]
#        self.info() # slow

    def info(self):
        ''' print info about numpy array
        very slow with large images '''
        y = self.arr
        out = f"{y.dtype}\t{str(y.shape)}\t<{y.min():.3f} \
            {y.mean():.3f} {y.max():.3f}> ({y.std():.3f})\t{type(y)} \
            bitdepth:{self.bitdepth} "
        print(out)
        return out

    @property
    def stats(self):
        ''' return stats dict
        statistics very slow with large images - disabled
        '''
        return {
            "name": self.name,
            "filetype": self.filetype,
            "bitdepth": self.bitdepth,
            "channels": self.channels,
            "size": f"{self.filesize/1024/1024: .3f} MB",
            "height": self.height,
            "width": self.width,
            "ratio": round(self.ratio, 2),
            "min": round(self.arr[self.slice].min(), 2),
            "max": round(self.arr[self.slice].max(), 2),
            "mean": round(self.arr[self.slice].mean(), 2),
            "std_dev": round(self.arr[self.slice].std(), 2),
        }

    def _get_bitdepth(self, arr):
        ''' read bitdepth before conversion to float '''
        if arr.dtype == np.uint8:
            return 8
        elif arr.dtype == np.uint16:
            return 16
        else:
            raise Exception(f"unsupported array type: {arr.dtype}")


# NUMPY TOOLS ====================================================

def im_rotate(y, angle=90):
    ''' rotate array by 90 degrees, k = number of rotations  '''
    k = angle // 90
    return np.rot90(y, -k, axes=(0, 1))


def np_to_pil(im):
    from PIL import Image
    ''' np float image (0..1) -> 8bit PIL image '''
    return Image.fromarray(img_as_ubyte(im))


def pil_to_np(im):
    ''' PIL image -> np float image (0..1) '''
    return img_as_float(im)


def blur(y, radius):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(y, radius)


def gray(y):
    if y.ndim == 2:
        return y
    elif y.ndim >= 3:
        return np.dot(y[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        raise Exception(f"gray conversion not supported, array ndim {y.ndim}")


def apply_gamma(f, gamma=1.):
    """gamma correction of an numpy float image, where gamma = 1. : no effect
        gamma > 1. : image will darken
        gamma < 1. : image will brighten"""
    return f ** gamma


def plti(im, name="", plot_axis=False, vmin=0, vmax=1, **kwargs):
    from matplotlib import pyplot as plt
    from PIL import Image
    ''' check image type and plot it '''
    if isinstance(im, Image.Image):  # PIL -> numpy
        im = pil_to_np(im)
    else:    # numpy image
        im = img_as_float(im)
    info(im, name)

    cmap = "gray" if im.ndim == 2 else "jet"

    plt.title(name)
    plt.imshow(im, interpolation="none",
               cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    if not plot_axis:
        plt.axis('off')  # turn off axis
    plt.show()


def numpy_to_png(im, fp_out,  bitdepth=8):
    from numpngw import write_png
    assert isinstance(im, (np.ndarray, np.generic))
    # ensure extenion
    Fp = Path(fp_out).with_suffix(".png")

    logging.info(f"saving array to png..{Fp}")
    # Image.fromarray(img_as_ubyte(im)).save(fp_out)
    write_png(Fp, im, bitdepth=bitdepth)
    logging.info("...saved")


def numpy_to_jpg(im, fp_out):
    import matplotlib.pyplot as plt
    im = img_as_float(im)
    Fp = Path(fp_out)

    logging.debug(f"saving array to jpg...{fp_out}")
    cmap = "gray" if im.ndim == 2 else "jet"

    # use matplotlib
    plt.imsave(Fp.with_suffix(".jpg"), im, cmap=cmap, vmin=0, vmax=1)
    logging.info(f"...saved: {fp_out}")


def load_image(fp):
    ''' load image from fp and return numpy float array (0..1) '''
    return img_as_float(imread(fp))  # convert to float - 0..1


def save_image(im, fp_out, bitdepth=None):
    from PIL import Image
    ''' float or uint16 numpy array --> 16 bit png
        uint8 numpy array --> 8bit png '''

    logging.info(f"saving image {fp_out}")
    Path(fp_out).parent.mkdir(exist_ok=True)

    if isinstance(im, Image.Image):
        im.save(fp_out)  # PIL save

    elif isinstance(im, np.ndarray):
        if bitdepth == 8:
            im = img_as_ubyte(im)
        if im.dtype in (np.uint8, "uint8"):
            numpy_to_png(im, fp_out)
        else:  # 16bit PNG - default output
            im = img_as_uint(im)  # accept float
            numpy_to_png(im, fp_out,  bitdepth=16)
    logging.info(f"image saved")
