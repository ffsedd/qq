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
from qq.ostools import to_trash
from qq.fileobj import File
from PIL import Image
from subprocess import run, check_output

FILETYPES = ['jpeg', 'bmp', 'png', 'tiff']
JPEGTRAN_EXE = 'jpegtran-droppatch'  # http://jpegclub.org/jpegtran/
MAGICK_EXE = 'convert'
EXIFTOOL_EXE = 'exiftool'


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
    def show(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.arr)
        plt.show()


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


class ImageFile(File):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def size(self):
        return Image.open(self).size

    def format(self):
        return Image.open(self).format

    def to_jpg(self, out_fp=None, trash=False, quality=85):
        logging.info(f"to_jpg {self}...")
        if self.format() == "JPEG":
            logging.info(f"... already in JPG format: {self.name} ")
            return
        if not out_fp:
            out_fp = self.with_suffix(".jpg")
        else:
            out_fp = Path(out_fp)
        Image.open(self).convert('RGB').save(out_fp, quality=quality)
        logging.info(f"...done: {self.name} ")

        if trash:
            to_trash(self)

    def to_png(self, out_fp=None, trash=False):
        logging.info(f"to_png {self}...")
        if self.format() == "PNG":
            logging.info(f"...already in PNG format: {self.name} ")
            return
        if not out_fp:
            out_fp = self.with_suffix(".png")
        else:
            out_fp = Path(out_fp)
        Image.open(self).save(out_fp)
        logging.info(f"...done: {self.name} ")
        if trash:
            to_trash(self)

    def to_png16(self, out_fp=None, trash=False):
        #        import numpy as np
        from numpngw import write_png
        import skimage.io
        if not out_fp:
            out_fp = self.with_suffix(".png")
        img = skimage.io.imread(self, plugin='tifffile')
        write_png(out_fp, img)

    def read_iptc_caption(self):
        cmd = f"{EXIFTOOL_EXE} -s3  -iptc:caption-abstract '{self}' "
        try:
            p = check_output(cmd, shell=True)
            self.iptc_caption = p.decode().strip()
        except Exception as e:
            logging.warn(f"exiftool error {cmd} {e}")
            return ""
        return self.iptc_caption

    def write_xmp_description(self, description):
        cmd = f"{EXIFTOOL_EXE} '{self}' -overwrite_original \
        -preserve -XMP-dc:Description='{description}' "
        r = run(cmd, shell=True)
        return r

    def write_iptc_keyword(self, tag):
        # append tag
        tag = tag.replace(" ", "_")
        cmd = f"{EXIFTOOL_EXE} '{self}' -overwrite_original \
        -preserve -iptc:keywords-={tag} -iptc:keywords+={tag} "
        r = run(cmd, shell=True)
        return r

    def rename_by_caption(self):
        try:
            cap = self.read_iptc_caption()
            if cap == "":
                return 0, "", None
            outfp = self.with_name(f"{cap}_#_{self.name}")
            self.rename(outfp)
            return 0, str(outfp), None  # rc, out, err

        except Exception as e:
            logging.critical("failed with exception: {}".format(e))
            return 1, str(self), e

    def update_exif_thumb(self, size=128):
        from qq.ostools import make_temp_dir
        # create thumbnail
        temp_fp = Path(make_temp_dir()) / \
            self.with_name(self.stem + '-thumb' + self.suffix)
        cmd = f'{MAGICK_EXE} "{self}" -quality 70 \
        -thumbnail {size}x{size} "{temp_fp}" '
        o = run(cmd, shell=True)
        if not temp_fp.is_file():
            logging.critical(
                "thumbnail not created, exit, code: {} \n {}".format(o, cmd))
            return 1, cmd, o
        # insert thumbnail
        cmd = f'{EXIFTOOL_EXE} -overwrite_original_in_place \
        "-thumbnailimage<={temp_fp}" "{self}" '
        o = run(cmd, shell=True)
        # remove temp file
        try:
            temp_fp.unlink()
        except Exception as e:
            logging.warning(e)
        return 0, o, None

    def sharpen(self, out_fp=None, radius=2.0,
                sigma=1.4, percent=120, threshold=3):
        logging.info(f"sharpen {self}...")
        if not out_fp:
            out_fp = self.with_name(self.stem + "_sharp" + self.suffix)
        unsharp = f"{radius:.1f}x{sigma:.1f}+\
        {percent/100:.2f}+{threshold/100:.2f}"
#        print ("unsharp mask:", unsharp)
        cmd = f"{MAGICK_EXE} -unsharp {unsharp} {self} {out_fp}"
        r = run(cmd, shell=True)
        out_image = ImageFile(out_fp)
        # write unsharp mask settings to XMP description
        out_image.write_xmp_description("USM_" + unsharp)
        out_image.write_iptc_keyword("sharp")    # add tag
        logging.info("...done: {self.name} ")
        return r

    def crop(self, geometry, out_fp):
        opt = r' -copy all -perfect -crop {0} "{1}" "{2}" '.format(
            geometry, self, out_fp)
    #    verbose = " -verbose"
        cmd = JPEGTRAN_EXE + opt  # + verbose
        p = run(cmd, shell=True)
        return p

    def rotate(self, angle):
        assert(angle % 90 == 0)
        logging.info(f"rotate  {self}...")
#        perfect = "-perfect"
        if self.format() == "JPEG":
            assert(angle in [90, 180, 270])
            perfect = ""
            cmd = f'{JPEGTRAN_EXE} -copy all {perfect} \
            -rotate {angle} -outfile "{self}" "{self}" '
            run(cmd, shell=True)
        else:
            Image.open(self).rotate(-angle).save(self)  # counterclockwise
        logging.info(f"...rotated: {self.name}")

    def get_exif(self):
        with Image.open(self) as img:
            img.verify()
            exif = img._getexif()
        return exif

    def resize(self, width, trash=False, quality=80):
        logging.info(f"resize {self}...")
        bak_fp = self.with_suffix(f"{self.suffix}0")
        self.rename(bak_fp)
        with Image.open(bak_fp) as img:
            if img.size[0] > width or img.size[1] > width:
                img.thumbnail((width, width))
                img.save(self, quality=quality)
                logging.info(f"...done: {self.name} ")
                if trash:
                    to_trash(bak_fp)
            else:
                logging.info(f"...image smaller than {width} px: {self.name} ")
                bak_fp.rename(self)

        return

    def to_trash(self):
        to_trash(self)


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

    if bitdepth == 8:
        im = img_as_ubyte(im) 
    elif bitdepth == 16:    
        im = img_as_uint(im)  # accept float
    else:
        raise Exception(f"unupported bitdepth {bitdepth}")   
        
    write_png(Fp, im, bitdepth=bitdepth)
    logging.debug(f"...saved: {fp_out}")


def numpy_to_jpg(im, fp_out):
    import matplotlib.pyplot as plt
    im = img_as_float(im)
    Fp = Path(fp_out)

    logging.debug(f"saving array to jpg...{fp_out}")
    cmap = "gray" if im.ndim == 2 else "jet"

    # use matplotlib
    plt.imsave(Fp.with_suffix(".jpg"), im, cmap=cmap, vmin=0, vmax=1)
    logging.debug(f"...saved: {fp_out}")


def load_image(fp):
    ''' load image from fp and return numpy float array (0..1) '''
    return img_as_float(imread(fp))  # convert to float - 0..1


def save_image(im, fp_out, bitdepth=None):
    from PIL import Image
    ''' float or uint16 numpy array --> 16 bit png
        uint8 numpy array --> 8bit png '''

    logging.info(f"saving image {fp_out}")
    fp = Path(fp_out)
    fp.parent.mkdir(exist_ok=True)
    
    # PIL image
    if isinstance(im, Image.Image):
        im.save(fp) 
    
    # numpy image
    elif isinstance(im, np.ndarray):
        print(fp.suffix.lower)
        if fp.suffix.lower in [".jpg",".jpeg"]:
            
            numpy_to_jpg(im, fp)
            
        if not bitdepth:
            if im.dtype in (np.uint8, "uint8"):
                bitdepth = 8
            else:  # 16bit PNG - default output
                bitdepth = 16

        numpy_to_png(im, fp,  bitdepth=bitdepth)
        
    else:
        raise Exception(f"unsupported image type {type(im)}")
