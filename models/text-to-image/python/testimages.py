#!/usr/bin/env python3

import sys
import os
import subprocess
import shutil
from shlex import quote
from argparse import ArgumentParser, Namespace
from typing import Sequence, Optional
from pathlib import Path



def get_arg_parser(argv: Optional[Sequence[str]] = None) -> ArgumentParser:
    parser = ArgumentParser()    
    parser.add_argument("--precision", choices = ["fp16", "fp32"],
                        help = "precision of imatge to be compare", 
                        type = str, default = "fp32")
    parser.add_argument("-i", "--image",
                        help = "Path to image to compare with golden",
                        type = str)
    parser.add_argument("-g", "--golden",
                        help = "Path to golden image",
                        type = str)
    return parser

def main(argv: Optional[Sequence[str]] = None):
    """Compare output image from test2image with cpu golden image"""
    parser = get_arg_parser()
    args = parser.parse_args(argv)

    #the absolute path of the directory where the program resides.
    pth = os.path.abspath(os.path.dirname(__file__))
    compare_images_script = os.path.join(pth, "compare_images.py")

    # Compare the generated image against the golden image generated at Cpu
    threshold = 0.60 if args.precision == "fp16" else 0.9

    cmd_compare = f"python3 {quote(compare_images_script)} --golden-img {quote(args.golden)} --test-img {quote(args.image)} -t {quote(str(threshold))}"

    print(cmd_compare)
    result_compare = subprocess.run(cmd_compare, capture_output=False, shell=True)
    if (result_compare.returncode != 0):
        sys.exit(f"Error in image comparison between generated ({image}) and golden image ({args.golden})")

    print("SUCCESS")
    
if __name__ == "__main__":
    sys.exit(main())
