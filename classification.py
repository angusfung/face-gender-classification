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
    Part 3: Create a classifier between Hader and Carell
    """)
parser.add_argument(
    '-o', '--optimal', 
    action='store_true',
    help="""
    Find the optimal parameters for gradient descent.
    Must also specify a part.
    """)
parser.add_argument(
    '-l', '--log-level',
    choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'],
    default='WARNING',
    help="set logging level (default: %(default)s)")
args = parser.parse_args()

# add file handler
def add_fh(name, level):
    if not os.path.isdir("logs"):
        os.makedirs("logs")
    # if log file already exists, delete it
    log_file = os.path.join("logs", name + ".log")
    if os.path.isfile(log_file):
        os.remove(log_file)
    # enhancement: move to Archive 
    fh = logging.FileHandler('logs/{}.log'.format(name))
    fh.setLevel(level)
    fh.setFormatter(logger_format)
    root.addHandler(fh)

def main():
    if args.download:
        add_fh('download', logging.DEBUG)
        act = [
            'Fran Drescher', 'America Ferrera', 'Kristin Chenoweth',
            'Alec Baldwin', 'Bill Hader', 'Steve Carell',
            'Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan',
            'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon',
            ]
        
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
        logger.info("Beginning Part 2 -- Generating training, test, validation set")
        act = [
            'Fran Drescher', 'America Ferrera', 'Kristin Chenoweth',
            'Alec Baldwin', 'Bill Hader', 'Steve Carell',
            ]
        make_dataset(act)

    if args.part == 3:
        logger.info("Beginning Part 3 -- Creating a classifier between Hader and Carell")
        act = ["hader", "carell"]
        
        if not args.optimal:
            theta = make_classifier(act, 'part3')
        else:
            theta = make_classifier(act, 'part3', True)
            
        actor_score = accuracy(act, 'test', theta)
        logger.info(actor_score)
        actor_score = accuracy(act, 'validation', theta)
        logger.info(actor_score)
            
    if args.part == 4:
        logger.info("Beginning Part 4 -- Visualizing theta")
        
        logger.info("Visualize theta with full training set")
        
        # load the saved theta
        if not os.path.isfile('part3.pkl'):
            raise ValueError("You must run part 3 first")
            
        with open(r'part3.pkl', 'rb') as f:
            theta = pickle.load(f)
        visualize(theta)
        
        logger.info("Visualize theta with two-image training set")
        act = ["hader", "carell"]
        theta = make_classifier(act, 'part4', training_size=2)
        visualize(theta)
        
    if args.part == 5:
        logger.info("Beginning Part 5 -- Gender Classification")
        training_act = [
            'Fran Drescher', 'America Ferrera', 'Kristin Chenoweth',
            'Alec Baldwin', 'Bill Hader', 'Steve Carell',
            ]
        test_act = [
            'Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan',
            'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon',
            ]
        gender_classification(training_act, test_act)
        
    if args.part == 6:
        logger.info("Beginning Part 6 -- Finite difference verification")
        verification()
        
    if args.part == 7:
        logger.info("Beginning Part 7 -- Multiple actor classification using one-hot")
        one_hot_classification()
                
        
if __name__ == '__main__':
    # set root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    
    # set module logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # use name to keep track of task
    logger_format = logging.Formatter('%(name)s - %(levelname)s: %(message)s')
    
    # add console handler to root logger (log INFO or above to terminal)
    # for general usage can set to WARN
    logger_stdout = logging.StreamHandler()
    log_level = getattr(logging, args.log_level)
    logger_stdout.setLevel(log_level)
    logger_stdout.setFormatter(logger_format)
    root.addHandler(logger_stdout)

    main()

