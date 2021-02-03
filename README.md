# BackgroundSubtractor4OCR
BackgroundSubtractor4OCR

## Information
This application uses multiple filter techniques to extract the background from the image to improve the OCR result. 

It is still very experimental and a good default setting isnt found yet. 
Please try it yourself.

## Installation
```
pip install -r requirements.txt
```

## Use 
```
python3 bs4ocr.py --PARAMETER fname
```

## Test
 ```
python3 bs4ocr.py ./test/test.png
```

## Parameter help
```
Subtract background for better OCR results.

positional arguments:
  fname                 filename of text file or path to files

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUTFOLDER, --outputfolder OUTPUTFOLDER
                        filename of the output
  -e EXTENSION, --extension EXTENSION
                        Extension of the img
  --extensionaddon EXTENSIONADDON
                        Addon to the fileextension
  -b BLURSIZE, --blursize BLURSIZE
                        Kernelsize for medianBlur
  -i BLURITER, --bluriter BLURITER
                        Iteration of the medianBlur
  -f, --fixblursize     Deactivate decreasing Blurkernelsize
  --blurfilter {Gaussian,Median}
                        Kernelsize for dilation
  -d DILSIZE, --dilsize DILSIZE
                        Kernelsize for dilation
  -s {cross,ellipse,rect}, --kernelshape {cross,ellipse,rect}
                        Shape of the kernel for dilation
  -c CONTRAST, --contrast CONTRAST
                        Higher contrast (experimental)
  -n, --normalize       Higher contrast (experimental)
  --normalize-only      Normalizes the image but doesnt subtract
  --normalize_auto      Auto-Normalization (experimental)
  --normalize_min NORMALIZE_MIN
                        Min value for background normalization
  --normalize_max NORMALIZE_MAX
                        Max value for background normalization
  --scale_channel {None,red,green,blue,cyan,magenta,yellow}
                        Shape of the kernel for dilation
  --scale_channel_value SCALE_CHANNEL_VALUE
                        Scale value
  --binarize            Use Adaptive-Otsu-Binarization
  --dpi DPI             Dots per inch (This value is used for binarization)
  -t, --textdilation    Deactivate extra dilation for text
  -q QUALITY, --quality QUALITY
                        Compress quality of the image like jpg
  -v, --verbose         show ignored files
```
