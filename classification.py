#!/usr/bin/env python
from download import *
import argparse
import logging
import os

parser = argparse.ArgumentParser(description='Choose Part')
parser.add_argument("--download", action = "store_true",
    help="Download FaceScrub dataset")

args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#logger.propagate = False

logger_format = logging.Formatter('%(levelname)s: %(message)s')

# add file handler
def add_handler(name):
    fh = logging.FileHandler('{}.log'.format(name))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logger_format)
    logger.addHandler(fh)

# add steam handler to std output 
# logger_info = logging.StreamHandler()
# logger_info.setLevel(logging.INFO)
# logger_info.setFormatter(logger_format)
# logger.addHandler(logger_stderr)


def main():
    if args.download:
        act = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
        os.makedirs("dataset/uncropped")
        os.makedirs("dataset/cropped")
        
        if (not os.path.isfile("facescrub_actors.txt") or 
                not os.path.isfile("facescrub_actresses.txt")):
            raise IOError("Either facescrub_actors.txt or facescrub_actresses.txt is missing. Please download it.") 
            
            # get_crop_pictures("facescrub_actors.txt", act)
            # get_crop_pictures("facescrub_actresses.txt",act)
            
    
    
if __name__ == '__main__':
    main()

