#!/usr/bin/env python
from download import *
import argparse
import logging
import os
import shutil

parser = argparse.ArgumentParser(description='Choose Part')
parser.add_argument("--download", action = "store_true",
    help="Download FaceScrub dataset")
parser.add_argument"("--

args = parser.parse_args()

# set root logger
root = logging.getLogger()
log_level = getattr(logging, loglevel.upper())
root.setLevel(log_level)

# set module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# use name to keep track of task
logger_format = logging.Formatter('%(name)s - %(levelname)s: %(message)s')

# add file handler
def add_fh(name):
    fh = logging.FileHandler('logs/{}.log'.format(name))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logger_format)
    root.addHandler(fh)

# add console handler to root logger (log INFO or above to terminal)
# for general usage can set to WARN
logger_stdout = logging.StreamHandler()
logger_stdout.setLevel(logging.INFO)
logger_stdout.setFormatter(logger_format)
root.addHandler(logger_stdout)


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

