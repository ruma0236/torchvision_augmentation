import glob
import os

paths = glob.glob('./*', recursive=True)

exts = ('superpixels')

# [os.remove(f) for f in glob.glob('./*.png', recursive=True)]

print(len(glob.glob('./*.jpg', recursive=True)))