import logging
import os
import random
import shutil

logger = logging.getLogger(__name__)

def make_dataset(act):
    
    makedirs('dataset/training')
    makedirs('dataset/validation')
    makedirs('dataset/test')
    
    # seed for reproducibility
    
    
    
    
    
def makedirs(dirs):
    """Make directory if doesn't exist, else delete existing directory"""
    if os.path.exists(dirs):
        shutil.rmtree(dirs)
        logger.warn("{} directory already exists, overwriting".format(dirs))
    else:
        os.makedirs(dirs)