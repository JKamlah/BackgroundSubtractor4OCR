#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

import cv2

# Command line arguments.
arg_parser = argparse.ArgumentParser(description='Subtract background for better OCR results.')
arg_parser.add_argument("fname", type=lambda x: Path(x), help="filename of text file or path to files", nargs='*')
arg_parser.add_argument("-o", "--outputfolder", default="./cleaned", help="filename of the output")
arg_parser.add_argument("-e", "--extension", default="jpg", help="Extension of the img")
arg_parser.add_argument("--extensionaddon", default=".prep", help="Addon to the fileextension")
arg_parser.add_argument("-b", "--blursize", default=59, type=int, help="Kernelsize for medianBlur")
arg_parser.add_argument("-d", "--dilsize", default=5, type= int, help="Kernelsize for dilation")
arg_parser.add_argument("-s", "--kernelshape", default="ellipse", help="Shape of the kernel for dilation", choices=["cross","ellipse","rect"])
arg_parser.add_argument("-c", "--contrast", action="store_true", help="Higher contrast (experimental)")
arg_parser.add_argument("-t", "--textdilation", action="store_false", help="Deactivate extra dilation for text")
arg_parser.add_argument("-q", "--quality", default=75, help="Compress quality of the image like jpg")
arg_parser.add_argument("-v", "--verbose", help="show ignored files", action="store_true")

args = arg_parser.parse_args()

def background_subtractor(img, dilsize=5, blursize=59, kernelshape="ellipse", textdilation=True, contrast=False, verbose=False):
    # Dilsize increasing makes scooping effects,
    # default (img, dilsize=19, blursize=21, contrast=0)
    img = cv2.imread(str(img), -1)
    rgb_planes = cv2.split(img)
    result_planes = []
    for idx, plane in enumerate(rgb_planes):
        dilated_img = plane
        kshape = {"rect":cv2.MORPH_RECT,"ellipse":cv2.MORPH_ELLIPSE,"cross":cv2.MORPH_CROSS}.get(kernelshape, cv2.MORPH_ELLIPSE)
        # Reduce influence of the text by dilation (round kernel produce atm the best results)
        if textdilation:
            dil_kernel = cv2.getStructuringElement(kshape, (dilsize, int(dilsize / 2)))
            dilated_img = cv2.dilate(plane, dil_kernel)
        dil_kernel = cv2.getStructuringElement(kshape, (dilsize, dilsize))
        dilated_img = cv2.dilate(dilated_img, dil_kernel)
        bg_img = cv2.medianBlur(dilated_img, blursize)

        if verbose:
            cv2.imwrite(f"Filtered_{idx}.jpg", bg_img)
            cv2.imwrite(f"Dilate_{idx}.jpg", dilated_img)
            
        # Slightly increase contrast (this option can lead to colorshifting-artefacts   
        if contrast:
            # Recommend a value higher than 75
            bg_img = cv2.multiply(bg_img, bg_img, dtype=cv2.CV_32F)
            bg_img = cv2.normalize(bg_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_8U)
       
        # Subtract bg from fg
        diff_img = 255 - cv2.absdiff(plane, bg_img)

        # Normalize the final image to the range 0-255
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        result_planes.append(norm_img)
    
    return cv2.merge(result_planes)

def main():
    # Set filenames or path
    if len(args.fname) == 1 and not args.fname[0].is_file():
        args.fname = list(Path(args.fname[0]).rglob(f"*.{args.extension}"))
    for img in args.fname:
        fout = Path(args.outputfolder).absolute().joinpath(img.name.rsplit(".", 1)[0] + f"{args.extensionaddon}.{args.extension}")
        print(fout)
        if not fout.parent.exists():
            fout.parent.mkdir()
        bg_sub = background_subtractor(img,dilsize=args.dilsize, blursize=args.blursize, kernelshape=args.kernelshape, textdilation=args.textdilation, contrast=args.contrast, verbose=args.verbose)
        if args.extension == "jpg":
            cv2.imwrite(str(fout.absolute()), bg_sub, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
        else:
            cv2.imwrite(str(fout.absolute()), bg_sub)

if __name__=="__main__":
    main()
