#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# Command line arguments.
arg_parser = argparse.ArgumentParser(description='Subtract background for better OCR results.')
arg_parser.add_argument("fname", type=lambda x: Path(x), help="filename of text file or path to files", nargs='*')
arg_parser.add_argument("-o", "--outputfolder", default="./cleaned", help="filename of the output")
arg_parser.add_argument("-e", "--extension", default="jpg", help="Extension of the img")
arg_parser.add_argument("--extensionaddon", default=".prep", help="Addon to the fileextension")
arg_parser.add_argument("-b", "--blursize", default=59, type=int, help="Kernelsize for medianBlur")
arg_parser.add_argument("-i", "--bluriter", default=1, type=int, help="Iteration of the medianBlur")
arg_parser.add_argument("-f", "--fixblursize", action="store_true", help="Deactivate decreasing Blurkernelsize")
arg_parser.add_argument("--blurfilter", default="Gaussian", type=str, help="Kernelsize for dilation",
                        choices=["Gaussian", "Median"])
arg_parser.add_argument("-d", "--dilsize", default=5, type=int, help="Kernelsize for dilation")
arg_parser.add_argument("-s", "--kernelshape", default="ellipse", type=str, help="Shape of the kernel for dilation",
                        choices=["cross", "ellipse", "rect"])
arg_parser.add_argument("-c", "--contrast", default=0.0, type=float, help="Higher contrast (experimental)")
arg_parser.add_argument("-n", "--normalize", action="store_true", help="Higher contrast (experimental)")
arg_parser.add_argument("--normalize-only", action="store_true", help="Normalizes the image but doesnt subtract")
arg_parser.add_argument("--normalize_auto", action="store_true", help="Auto-Normalization (experimental)")
arg_parser.add_argument("--normalize_min", default=0, type=int, help="Min value for background normalization")
arg_parser.add_argument("--normalize_max", default=255, type=int, help="Max value for background normalization")
arg_parser.add_argument("--scale_channel", default="None", type=str, help="Shape of the kernel for dilation",
                        choices=["None", "red", "green", "blue", "cyan", "magenta", "yellow"])
arg_parser.add_argument("--scale_channel_value", default=0.0, type=float, help="Scale value")
arg_parser.add_argument("--binarize", action="store_true", help="Use Adaptive-Otsu-Binarization")
arg_parser.add_argument("--dpi", default=300, type=int, help="Dots per inch (This value is used for binarization)")
arg_parser.add_argument("-t", "--textdilation", action="store_false", help="Deactivate extra dilation for text")
arg_parser.add_argument("-q", "--quality", default=100, help="Compress quality of the image like jpg")
arg_parser.add_argument("-v", "--verbose", help="show ignored files", action="store_true")

args = arg_parser.parse_args()

def channelscaler(channel, value):
    channel = cv2.multiply(channel, value)
    channel = np.where(channel < 255, 255, channel)
    return channel


# -i 4 -b 150 -d 10  good settings atm for 300 dpi
def subtractor(img, dilsize: int = 15, blursize: int = 59, kernelshape: str = "ellipse",
               bluriter: int = 1, fix_blursize: bool = False, blurfilter: str = "Gaussian",
               textdilation: bool = True, contrast: bool = False, verbose: bool = False):
    """
    The text in the image will be removed, the background smoothed and than extracted from the original image
    :param img:
    :param dilsize:
    :param blursize:
    :param kernelshape:
    :param normalize:
    :param norm_min:
    :param norm_max:
    :param norm_auto:
    :param bluriter:
    :param fix_blursize:
    :param blurfilter:
    :param textdilation:
    :param contrast:
    :param verbose:
    :return:
    """
    rgb_planes = cv2.split(img)
    result_planes = []

    # Only odd blurkernelsize are valid
    blursize = blursize + 1 if blursize % 2 == 0 else blursize

    for idx, plane in enumerate(rgb_planes[:3]):
        dilated_img = plane
        kshape = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}.get(kernelshape,
                                                                                                      cv2.MORPH_ELLIPSE)
        # Reduce influence of the text by dilation (round kernel produce atm the best results)
        if textdilation:
            dil_kernel = cv2.getStructuringElement(kshape, (int(dilsize / 2), dilsize))
            dilated_img = cv2.dilate(plane, dil_kernel, iterations=3)
            dil_kernel = cv2.getStructuringElement(kshape, (int(dilsize / 2) + 1, dilsize + 1))
            dilated_img = cv2.erode(dilated_img, dil_kernel, iterations=1)
        else:
            dil_kernel = cv2.getStructuringElement(kshape, (dilsize, dilsize))
            dilated_img = cv2.dilate(dilated_img, dil_kernel)

        bg_img = dilated_img
        for ksize in np.linspace(blursize, 1, num=bluriter):
            if not fix_blursize:
                if blurfilter == "Gaussian":
                    bg_img = cv2.GaussianBlur(bg_img,
                                              (int(ksize) + (1 + int(ksize) % 2), int(ksize) + (1 + int(ksize) % 2)), 0)
                else:
                    bg_img = cv2.medianBlur(bg_img, (int(ksize) + (1 + int(ksize) % 2)))
            else:
                if blurfilter == "Gaussian":
                    bg_img = cv2.GaussianBlur(bg_img, (blursize, blursize), 0)
                else:
                    bg_img = cv2.medianBlur(bg_img, blursize)

        if verbose:
            cv2.imwrite(f"Filtered_{idx}.jpg", bg_img)
            cv2.imwrite(f"Dilate_{idx}.jpg", dilated_img)

        # Subtract bg from fg
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # Increases the contrast
        if contrast:
            diff_img = cv2.add(norm_img, plane * contrast, dtype=cv2.CV_8U)
            # Normalize the final image to the range 0-255
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        result_planes.append(norm_img)

    return cv2.merge(result_planes)


def normalizer(img, norm_min: int = 0, norm_max: int = 255, norm_auto: bool = False):
    """
    Normalizes the histogram of the image
    :param img: path object of the image
    :param norm_min: max min value
    :param norm_max: min max value
    :param auto: auto normalizer
    :return:
    """
    rgb_planes = cv2.split(img)
    result_planes = []

    for idx, plane in enumerate(rgb_planes[:3]):
        if norm_auto:
            auto_min = np.min(np.where((norm_min <= 25, 255)))
            auto_max = np.max(np.where((norm_min <= 220, 0)))
            plane = np.where(plane <= auto_min, auto_min, plane)
            plane = np.where(plane >= auto_max, auto_max, plane)
        else:
            plane = np.where(plane <= norm_min, norm_min, plane)
            plane = np.where(plane >= norm_max, norm_max, plane)
        norm_img = cv2.normalize(plane, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        result_planes.append(norm_img)

    return cv2.merge(result_planes)


def main():
    # Set filenames or path
    if len(args.fname) == 1 and not args.fname[0].is_file():
        args.fname = list(Path(args.fname[0]).rglob(f"*.{args.extension}"))
    for fname in args.fname:
        print(fname.name + " in process!")
        try:
            # Try to get dpi information
            from PIL import Image
            dpi = Image.open(fname).info['dpi']
            args.dpi = np.mean(dpi, dtype=int)
            print("DPI was set to:", args.dpi)
        except:
            pass

        # Read image
        img = cv2.imread(str(fname), -1)
        resimg = img
        # Channel scaler
        if args.scale_channel != 'None' and len(img.shape) > 2:
            if args.scale_channel in ['red', 'yellow', 'magenta']:
                img[:, :, 0] = channelscaler(img[:, :, 0], args.scale_channel_value)
            if args.scale_channel in ['green', 'yellow', 'cyan']:
                img[:, :, 1] = channelscaler(img[:, :, 1], args.scale_channel_value)
            if args.scale_channel in ['blue', 'magenta', 'cyan']:
                img[:, :, 2] = channelscaler(img[:, :, 2], args.scale_channel_value)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Background normalizer
        if args.normalize or args.normalize_only:
            img = normalizer(img, args.normalize_min, args.normalize_max,  args.normalize_auto)
        # Background subtractor
        if not args.normalize_only:
            resimg = subtractor(img, dilsize=args.dilsize, blursize=args.blursize, kernelshape=args.kernelshape,
                                bluriter=args.bluriter, fix_blursize=args.fixblursize,
                                textdilation=args.textdilation, contrast=args.contrast, verbose=args.verbose)
        # Image binarizer
        if args.binarize:
            DPI = args.dpi + 1 if args.dpi % 2 == 0 else args.dpi
            resimg = resimg if len(resimg.shape) == 2 else cv2.cvtColor(resimg, cv2.COLOR_BGR2GRAY)
            resimg = cv2.adaptiveThreshold(resimg, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                           cv2.THRESH_BINARY, DPI, int(DPI / 12))
            args.extensionaddon = args.extensionaddon + ".bin"
        # Output
        fout = Path(args.outputfolder).absolute().joinpath(
            fname.name.rsplit(".", 1)[0] + f"{args.extensionaddon}.{args.extension}")
        if not fout.parent.exists():
            fout.parent.mkdir()
        if args.extension == "jpg":
            cv2.imwrite(str(fout.absolute()), resimg, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
        else:
            cv2.imwrite(str(fout.absolute()), resimg)
        print(str(fout) + " created!")


if __name__ == "__main__":
    main()
