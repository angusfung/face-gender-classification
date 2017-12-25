import logging
import os
import random
import shutil
from shutil import copyfile
from scipy.misc import imread
import numpy as np

logger = logging.getLogger(__name__)


# part 2

def make_dataset(act, training_size=100, validation_size=10, test_size=10):
                
    makedirs('dataset/training')
    makedirs('dataset/validation')
    makedirs('dataset/test')
    
    # seed for reproducibility
    random.seed(0)
    
    # list of picture names
    pic_names = os.listdir('dataset/cropped')
    # get number of pictures
    size = len(pic_names)

    # create non-overlapping training, validation and test set
    # 100, 10, 10 per actor, respectively
    
    for actor in act:
        actor = actor.split()[1].lower()
        actor_start, actor_end = get_range(actor, 'dataset/cropped')
        actor_size = actor_end - actor_start
        # generate a set of random numbers
        rand = random.sample(range(actor_start, actor_end), actor_size)
        
        if actor_size < training_size + validation_size + test_size:
            raise ValueError("""
            Please choose a smaller training|validation|test size as it exceeds {}, the number of 
            pictures of actor {}
            """.format(actor_size, actor))
            
        for i in range(training_size + validation_size + test_size):
            pic_name = pic_names[rand[i]]
            pic_path = os.path.join('dataset/cropped', pic_name)
            if i < training_size:
                copyfile(pic_path, os.path.join('dataset/training', pic_name)) 
                logger.debug("Training Picture {} was added as {}".format(i+1, pic_name))
            elif (i >= training_size) and (i < training_size + validation_size):
                copyfile(pic_path, os.path.join('dataset/validation', pic_name)) 
                logger.debug("Validation Picture {} was added as {}".format(i+1, pic_name))
            else:
                copyfile(pic_path, os.path.join('dataset/test', pic_name)) 
                logger.debug("Test Picture {} was added as {}".format(i+1, pic_name))

# part 3

def make_classifier(act):
    x = np.empty((1024, 0), int)
    y = np.empty((0, 1), int)
    
    for actor in act:
        # check if dataset exists, else raise exception
        if not os.path.exists('dataset/training'):
            raise ValueError('Your training dataset is empty')
                        
        start, end = get_range(actor, 'dataset/training')
        pic_names = os.listdir('dataset/training')
        
        # generate X and y matrix (theta^T X - Y)
        """
        Variable          Dimensions
        ----------------------------
        X                 (1024L, 100L)
        theta             (1024L, 1L)
        theta^T X         (100L, )
        Y                 (100L, )
        """
        
        # +1 for exclusive range
        for index in range(start, end + 1):
            
            # True to keep image as grey scale, scipy opens to RGB by default
            image = imread(os.path.join('dataset/training', pic_names[index]), True)
            
            # flatten to column vector (1024 by 1)
            im = np.reshape(image, (1024, 1))
            x = np.hstack((x, im))
            
            if actor == "hader":
                y = np.vstack((y, 1))
            elif actor == "carell":
                y = np.vstack((y, 0))
            else:
                raise ValueError("Unrecognized actor name")
        
    # reshape (200L, 1L) to (200L, )
    y = np.reshape(y, (200))

# helper functions

def makedirs(dirs):
    """Make directory if doesn't exist, else delete existing directory"""
    if os.path.exists(dirs):
        shutil.rmtree(dirs)
        logger.warn("{} directory already exists, overwriting".format(dirs))
    else:
        os.makedirs(dirs)

def get_range(actor, directory):
    """Return start and end range of the actor in the filesystem"""
    start_found = 0
    
    for num, filename in enumerate(os.listdir(directory)):
        if actor in filename and start_found == 0:
            start = num
            start_found = 1
        if actor not in filename and start_found == 1:
            end = num - 1
            return start, end
    # check if start is defined but end is not
    # it means the actor is last in the filesystem
    if start:
        end = num
        return start, end        

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

