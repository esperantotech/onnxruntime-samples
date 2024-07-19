#!/usr/bin/env python3

import os 
import sys
from argparse import ArgumentParser, Namespace
from typing import Sequence, Optional

def check_positive(value):
    try:
        value = int(value)
        if value <= 0:
            raise argparse.ArgumentTypeError(f'{value} is not a positive integer.')
    except ValueError:
        raise Exception(f'{value} is not an integer.')
    return value

def get_arg_parser(argv: Optional[Sequence[str]] = None) -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("-a", "--artifacts", default = "../../../DownloadArtifactory")
    parser.add_argument("-v", "--verbose", action = "store_true",
                        help = "It shows help info messages")
    parser.add_argument("-b", "--batch", 
                        help = "specify the number of batches", 
                        type = check_positive, choices = range(1, 10000), default = 1)
    parser.add_argument("-t", "--totalInferences", 
                        help = "Defines the total number of inference to do",
                        type = check_positive, default = 20)
    parser.add_argument("-s", "--silicon", action = "store_true",
                        help = "Define silicon device to be used")
    parser.add_argument("-p", "--prompt",
                        help = "description text about the image to generate",
                        type = str, default = "an astronaut riding a horse")
    parser.add_argument("-d", "--datatype", choices = ["fp16", "fp32"],
                        help = "specify the model data type",
                        type = str, default = "fp32")
    return parser
