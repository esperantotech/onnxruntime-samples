#!/usr/bin/env python3

import os
import argparse
import sys
from image_similarity_measures.evaluate import evaluation

# Parsing arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description = "Test generated image against golden reference")

    parser.add_argument('--golden-img', metavar = "STR", type = str, default = None,
                        help = 'Path to the golden image')
    parser.add_argument('--test-img', metavar = "STR", type = str, default = None,
                        help = 'Path to the generated image')
    parser.add_argument("-m", '--metric', metavar = "STR", type = str, default = 'fsim',
                        help = 'The metric to use to compare images')
    parser.add_argument("-t", '--threshold', type = float, default = '0.8',
                        help = 'The threshold value used in a metric to give a match')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    metric = args.metric
    threshold = args.threshold 
    
    metric_list = ["fsim", "issm", "psnr", "rmse", "sam", "sre", "ssim", "uiq"]
    if metric not in metric_list:
        sys.exit(f"Error: The metric {metric} is not implemented. The supported metrics are: {metric_list}.")    

    result = evaluation(org_img_path=args.golden_img, pred_img_path=args.test_img, metrics=[metric])
    if metric not in result:
        sys.exit(f"Error: The evaluation of selected metric {metric} failed.")

    if result[metric]<threshold:
        sys.exit(f"Error: The images missmatch. The similarity metric result is {result}, which is below defined threshold {threshold}.")

    print(f"The images {args.golden_img} and {args.test_img} match. The similarity metric result is {result}.")

if __name__ == "__main__":
    sys.exit(main())
