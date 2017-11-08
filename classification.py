#!/usr/bin/env python
from download import *
import argparse
import logging
import os
import shutil

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

# add steam handler to std output (log only INFO or above to std out)
# could alternatively log stderr
logger_stdout = logging.StreamHandler()
logger_stdout.setLevel(logging.INFO)
logger_stdout.setFormatter(logger_format)
logger.addHandler(logger_stdout)


def main():
    if args.download:
        act = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
        
        if os.path.isdir("dataset"):
            logger.warn("dataset directory already exists, removed existing")
            shutil.rmtree("dataset")
        os.makedirs("dataset/uncropped")
        os.makedirs("dataset/cropped")
        
        if (not os.path.isfile("facescrub_actors.txt") or 
                not os.path.isfile("facescrub_actresses.txt")):
            raise IOError("Either facescrub_actors.txt or facescrub_actresses.txt is missing. Please download it.") 
            
        get_crop_pictures("facescrub_actors.txt", act)
            # get_crop_pictures("facescrub_actresses.txt",act)
            
    
    
if __name__ == '__main__':
    main()

