import logging
import os
import random
import shutil

logger = logging.getLogger(__name__)

def make_dataset(act, training_size=100, validation_size=10, test_size=10):
    def get_range(actor):
        """Return start and end range of the actor in the filesystem"""
        start_found = 0
        
        for num, filename in enumerate(os.listdir('dataset/cropped')):
            if actor in filename and start_found == 0:
                start = num
                start_found = 1
            if actor not in filename and start_found == 1:
                end = num - 1
                return start, end
                
    makedirs('dataset/training')
    makedirs('dataset/validation')
    makedirs('dataset/test')
    
    # seed for reproducibility
    random.seed(0)
    
    # list of picture names
    pic_names = os.listdir('dataset/cropped')
    # get number of pictures
    size = len(pic_names)
    # name = a.split()[1].lower()
    for actor in act:
        actor_start, actor_end = get_range(actor)
        actor_size = actor_end - actor_start
        # generate a set of random numbers
        rand = random.sample(range(act_start, actor_end), actor_size)
        
        if actor_size < training_size + validation_size + test_size:
            raise ValueError("""
            Please choose a smaller training|validation|test size as it exceeds {}, the number of 
            pictures of actor {}
            """.format(actor_size, actor))
            
        # heed attention to strictness of equality
        
        for i in range(training_size + validation_size + test_size):
            if i < training_size:
                logger.warn("Training Pic {} is {}".format(i, pic_names[i]))
            elif (i > training_size) and (i < training_size + validation_size):
               logger.warn("Validation Pic {} is {}".format(i, pic_names[i]))
            else:
                logger.warn("Test Pic {} is {}".format(i, pic_names[i]))
    
def makedirs(dirs):
    """Make directory if doesn't exist, else delete existing directory"""
    if os.path.exists(dirs):
        shutil.rmtree(dirs)
        logger.warn("{} directory already exists, overwriting".format(dirs))
    else:
        os.makedirs(dirs)