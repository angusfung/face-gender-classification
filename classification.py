#!/usr/bin/env python2
from download import *
import argparse
import logging

parser = argparse.ArgumentParser(description='Choose Part')
parser.add_argument("--download", action = "store_true",
    help="Download FaceScrub dataset")

args = parser.parse_args()

logger = logging.getLogger('classification')
logger.setLevel(logging.DEBUG)
logger.propagate = False

logger_format = logging.Formatter('%(levelname)s: %(message)s')

# add file handler
fh = logging.FileHandler('classification.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logger_format)
logger.addHandler(fh)

# add steam handler to std output (change INFO to WARNING if necessary)
logger_info = logging.StreamHandler()
logger_info.setLevel(logging.INFO)
logger_info.setFormatter(logger_format)
logger.addHandler(logger_stderr)


def main():
    
    
if __name__ == '__main__':
    main()

