#!/usr/bin/env python
from download import *
from src import *
import argparse
import logging
import os
import shutil

parser = argparse.ArgumentParser(description='Choose Part')
parser.add_argument(
    '-d', '--download', 
    action='store_true',
    help="Download FaceScrub dataset")
parser.add_argument(
    '-p', '--part',
    type=int,
    help="""
    Specify the part:
    Part 2: Create non-overlapping training, validation, and test set
    DEFAULT: 100, 10, 10
    """)
parser.add_argument(
    '-l', '--log-level',
    choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'],
    default='WARNING',
    help="set logging level (default: %(default)s)")
args = parser.parse_args()

# set root logger
root = logging.getLogger()
root.setLevel(logging.DEBUG)

# set module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# use name to keep track of task
logger_format = logging.Formatter('%(name)s - %(levelname)s: %(message)s')

# add file handler
def add_fh(name):
    if not os.path.isdir("logs"):
        os.makedirs("logs")
    # if log file already exists, delete it
    log_file = os.path.join("logs", name + ".log")
    if os.path.isfile(log_file):
        os.remove(log_file)
    # enhancement: move to Archive 
    fh = logging.FileHandler('logs/{}.log'.format(name))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logger_format)
    root.addHandler(fh)

# add console handler to root logger (log INFO or above to terminal)
# for general usage can set to WARN
logger_stdout = logging.StreamHandler()
log_level = getattr(logging, args.log_level)
logger_stdout.setLevel(log_level)
logger_stdout.setFormatter(logger_format)
root.addHandler(logger_stdout)

def main():
    if args.download:
        add_fh('download')
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
        get_crop_pictures("facescrub_actresses.txt",act)
    
    if args.part == 2:
        logger.info("Beginning Part 2")
        act = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
        make_dataset(act)
        
if __name__ == '__main__':
    main()

