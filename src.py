import logging
import os
import random
import shutil

logger = logging.getLogger(__name__)

def make_dataset(act):
    def get_range(actor):
        """Return start and end range of the actor in the filesystem"""
        start_found = 0
        
        for num, filename in enumerate(os.listdir('dataset/cropped')):
            if actor in filename and start_found == 0:
                start = num
                start_found = 1
            if actor not in filename and start_found == 1:
                end = num
                return start, end
                
    makedirs('dataset/training')
    makedirs('dataset/validation')
    makedirs('dataset/test')
    
    # seed for reproducibility
    random.seed(0)
    
    # get number of pictures
    size = len(os.listdir('dataset/cropped'))
    
    print(get_range('baldwin'))
    print(get_range('carell'))
    
    
    # for actor in act:
    #     i = 0
    #     while i < 100:
    #         index = random.randint(0, size)
    #         if a in 
    # 
    
def makedirs(dirs):
    """Make directory if doesn't exist, else delete existing directory"""
    if os.path.exists(dirs):
        shutil.rmtree(dirs)
        logger.warn("{} directory already exists, overwriting".format(dirs))
    else:
        os.makedirs(dirs)