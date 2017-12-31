import logging
import os
import random
import shutil
from shutil import copyfile
from scipy.misc import imread
import numpy as np
from classification import *
import pickle
from pylab import imshow, show

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

def make_classifier(act, optimal=False):
    x = np.empty((1024, 0), int)
    y = np.empty((0, 1), int)
    
    for actor in act:
        # check if dataset exists, else raise exception
        if not os.path.exists('dataset/training'):
            raise ValueError('Your training dataset is empty')
                        
        start, end = get_range(actor, 'dataset/training')
        pic_names = os.listdir('dataset/training')
        
        # generate X and Y matrix (theta^T X - Y)
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
    
    # find optimal parameters
    if optimal:
        optimal_params(act, x, y)
        return
        
    # run gradient descent
        
    init_theta = np.array([0.00] * 1025)
    theta, f_value = grad_descent(f, df, x, y, init_theta, 1e-11)      
    actor_score = accuracy(act, 'test', theta)
    logger.info(actor_score)
    actor_score = accuracy(act, 'validation', theta)
    logger.info(actor_score)
    
    # save the theta value
    
    file = open(r'part3.pkl', 'ab')
    pickle.dump(theta, file)
    file.close()

# part 4

def visualize(theta):
    """visualize theta"""
    # ignore the bias term
    im = np.reshape(theta[1:], (32, 32))
    imshow(im)
    show()
    
def two_image_classifier(act):
    x = np.empty((1024, 0), int)
    y = np.empty((0, 1), int)
    
    for actor in act:
        # check if dataset exists, else raise exception
        if not os.path.exists('dataset/training'):
            raise ValueError('Your training dataset is empty')
                        
        start, end = get_range(actor, 'dataset/training')
        pic_names = os.listdir('dataset/training')
        
        # generate X and Y matrix (theta^T X - Y)

        for index in range(start, start + 2):
            
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
    y = np.reshape(y, (4))
    
    # run gradient descent
        
    init_theta = np.array([0.00] * 1025)
    theta, f_value = grad_descent(f, df, x, y, init_theta, 1e-11)      
    actor_score = accuracy(act, 'test', theta)
    logger.info(actor_score)
    actor_score = accuracy(act, 'validation', theta)
    logger.info(actor_score)
    visualize(theta)
    
# --------------------- helper functions --------------------- #
    
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

def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-10   
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 40000
    iter  = 0
    while (np.linalg.norm(t - prev_t) >  EPS and iter < max_iter):
        prev_t = t.copy()
        t -= alpha * df(x, y, t)
        if iter % 2000 == 0:
            
            logger.debug("Iteration {}".format(iter))
            logger.debug("x = ({:.2f}, {:.2f}, {:.2f}, ...,) f(x)={:.2f})".format(
                t[0], t[1], t[2], f(x, y, t)))
            logger.debug("Gradient: {} \n".format(df(x, y, t)))
        iter += 1
    return (t,f(x, y, t))        
    
def f(x, y, theta):
    """cost function"""
    x = np.vstack((np.ones((1, x.shape[1])), x))
    return np.sum((y - np.dot(theta.T,x)) ** 2)

def df(x, y, theta):
    """derivative of cost function used in gradient descent"""
    x = np.vstack((np.ones((1, x.shape[1])), x))
    return -2 * np.sum((y - np.dot(theta.T, x)) * x, 1)  

def h(x, opt_theta):
    """dot product of x, optimal theta"""
    return np.dot(x, opt_theta)

def accuracy(actor_list, dataset, opt_theta):
    """test accuracy of theta on the validation and test set"""
    actor_score = {actor: 0 for actor in actor_list}
    
    for actor in actor_list:
        if not os.path.exists(os.path.join('dataset', dataset)):
            raise ValueError("The directory {} does not exist under dataset/".format(dataset))
            
        start, end = get_range(actor, os.path.join('dataset', dataset))
        pic_names = os.listdir(os.path.join('dataset', dataset))
        
        for index in range(start, end + 1):
            # True to keep image as grey scale, scipy opens to RGB by default
            image = imread(os.path.join(os.path.join('dataset', dataset), pic_names[index]), True)
            
            # flatten to column vector (1024L,)
            im = np.reshape(image, (1024))
            
            # add the bias
            im = np.hstack((1, im))
            
            h_ = h(im, opt_theta)
            
            if h_ >= 0.5 and (actor == 'hader'):
                actor_score['hader'] += 1
            elif h_ < 0.5 and (actor == 'carell'):
                actor_score['carell'] += 1
    
    return actor_score

    
def optimal_params(act, x, y):
    """find optimal parameters"""
    alpha_values = [1e-7, 1e-8, 1e-9, 1e-10, 1e-11] 
    initial_theta = [i * 0.01 for i in range (-10, 10, 2)]
    
    # add file handler
    add_fh('optimal_theta', logging.INFO)
    
    for alpha in alpha_values:
        for theta in initial_theta:
            logger.info("Gradient descent using alpha {} and theta {}".format(alpha, theta))
            theta_ = np.array([theta] * 1025)
            opt_theta, value = grad_descent(f, df, x, y, theta_, alpha)
            actor_score_val = accuracy(act, 'validation', opt_theta)
            actor_score_test = accuracy(act, 'test', opt_theta)
            logger.info("Validation Score: {}".format(actor_score_val))
            logger.info("Test Score: {}".format(actor_score_test))
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

