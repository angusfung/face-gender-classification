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
import matplotlib.pyplot as plt

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

def make_classifier(
    act, part_name, optimal=False,
    training_size=100):
        
    x = np.empty((1024, 0), int)
    y = np.empty((0, 1), int)
    
    if args.part == 7:
        y = np.empty((6, 0), int)
    
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
            if (index - start >= training_size):
                break
            
            # True to keep image as grey scale, scipy opens to RGB by default
            image = imread(os.path.join('dataset/training', pic_names[index]), True)
            
            # flatten to column vector (1024 by 1)
            im = np.reshape(image, (1024, 1))
            x = np.hstack((x, im))
            
            # if part 3, do hader VS. carell
            if args.part == 3:
                if actor == "hader":
                    y = np.vstack((y, 1))
                elif actor == "carell":
                    y = np.vstack((y, 0))
                else:
                    raise ValueError("Unrecognized actor name")
            
            # if part 5, do gender classification
            elif args.part == 5:
                if actor in ['drescher', 'ferrera', 'chenoweth']:
                    y = np.vstack((y, 1))
                elif actor in ['baldwin', 'hader', 'carell']:
                    y = np.vstack((y, 0))
                else:
                    raise ValueError("Unrecognized actor name")
                
            # if part 7, do one-hot classification
            elif args.part == 7:
                if actor == "drescher":
                    y = np.hstack((y, np.reshape([1,0,0,0,0,0],(6,1))))
                elif actor == "ferrera":
                    y = np.hstack((y, np.reshape([0,1,0,0,0,0],(6,1))))
                elif actor == "chenoweth":
                    y = np.hstack((y, np.reshape([0,0,1,0,0,0],(6,1))))
                elif actor == "baldwin":
                    y = np.hstack((y, np.reshape([0,0,0,1,0,0],(6,1))))
                elif actor == "hader":
                    y = np.hstack((y, np.reshape([0,0,0,0,1,0],(6,1))))
                elif actor == "carell":
                    y = np.hstack((y, np.reshape([0,0,0,0,0,1],(6,1))))
                else:
                    raise ValueError("Unrecognized actor name")
                    
    # reshape (200L, 1L) to (200L, )
    total_training_size = training_size * len(act)
    if not args.part == 7:
        y = np.reshape(y, (total_training_size))
    
    # find optimal parameters
    if optimal:
        optimal_params(act, x, y)
        return
        
    # run gradient descent
    if args.part == 7:
        init_theta = np.array([0.00] * 1025 * 6)
        init_theta = np.reshape(init_theta, (1025, 6))
        theta, f_value = grad_descent(fv, dfv, x, y, init_theta, 1e-11)  
    else:
        init_theta = np.array([0.00] * 1025)
        theta, f_value = grad_descent(f, df, x, y, init_theta, 1e-11)     
    
    # save the theta value
    
    file = open(r'{}.pkl'.format(part_name), 'ab')
    pickle.dump(theta, file)
    file.close()
    
    return theta

# part 4

def visualize(theta):
    """visualize theta"""
    # ignore the bias term
    im = np.reshape(theta[1:], (32, 32))
    imshow(im)
    show()
    
# part 5 

def gender_classification(training_act, test_act):
    
    # vary training size
    max_size = 70
    training_size = [10 * x for x in range(1, max_size / 10 + 1)]
    
    theta_dict = {}
    validation_dict = {}
    training_dict = {}
    
    make_dataset(training_act + test_act, training_size=70, validation_size=10, test_size=10)
    
    training_act = [name.split()[1].lower() for name in training_act]
    test_act = [name.split()[1].lower() for name in test_act]
    
    # train the classifier
    for size in training_size:
        logger.info("Gender classification with training size {}".format(size))
        theta = make_classifier(training_act, 'part5', training_size=size)
        theta_dict[size] = theta
        
    # test the classifier
    for size, theta in theta_dict.iteritems():
        
        # test on different actors
        male_list = ['butler', 'radcliffe', 'vartan'] 
        female_list = ['bracco', 'gilpin', 'harmon']
        
        validation_score = accuracy(test_act, 'validation', theta, 10, female_list, male_list)
        validation_dict[size] = validation_score
        
        # test on same actors
        male_list = ['baldwin', 'hader', 'carell']
        female_list = ['drescher', 'ferrera', 'chenoweth']
        
        training_score = accuracy(training_act, 'validation', theta, 10, female_list, male_list)
        training_dict[size] = training_score
            
    # plot
    
    # sort the dictionary by key (e.g 10, 20) and return as list of tuples
    validation_dict = sorted(validation_dict.items())
    training_dict = sorted(training_dict.items())
    
    # unpack list of tuples into two tuples
    size, validation_score = zip(*validation_dict)
    size, training_score = zip(*training_dict)
    
    plt.plot(size, validation_score, label = 'Validation Score')
    plt.plot(size, training_score, label = 'Training Score')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score (%)')
    plt.title('Performance vs. Size')
    plt.legend(['Validation Score', 'Training Score'], loc=7)
    plt.savefig("PerformanceVsSize.png")
    plt.show()
    
# part 6 
def verification():
    """sanity check of dfv using finite differences"""

    # create random x and y matrices
    random.seed(0)
    y = np.reshape(np.random.rand(4 * 200), (4, 200))
    x = np.reshape(np.random.rand(1024 * 200), (1024, 200))
    theta = np.reshape(np.random.random(1025 * 4), (1025, 4))
    
    # initialize finite difference
    h = 1e-9
    
    for i, j in [(0,0), (0,1), (1,0), (1,1)]:
        dtheta = np.zeros((1025, 4))
        dtheta[i,j] = h
        logger.info("Finite difference for coordinate ({},{})".format(i, j))
        logger.info((fv(x, y, theta + dtheta) - fv(x, y, theta - dtheta)) / (2 * h))
        logger.info(dfv(x, y, theta))
        logger.info("---------------------------------------------")
    
# part 7
def one_hot_classification():
    act = ['drescher', 'ferrera', 'chenoweth', 'baldwin', 'hader', 'carell']
    theta = make_classifier(act, 'part7')
        
    # save the theta value
    file = open(r'part{}.pkl'.format(args.part), 'ab')
    pickle.dump(theta, file)
    file.close()
    
    actor_score = accuracy(act, 'training', theta)
    logger.info("Performance on the training set is {}".format(actor_score))
    actor_score = accuracy(act, 'test', theta)
    logger.info("Performance on the testing set is {}".format(actor_score))
    actor_score = accuracy(act, 'validation', theta)
    logger.info("Performance on the validation set is {}".format(actor_score))
    
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
    logger.info("Beginning Gradient Descent")
    EPS = 1e-10   
    prev_t = init_t-10 * EPS
    t = init_t.copy()
    max_iter = 40000
    iter  = 0
    while (np.linalg.norm(t - prev_t) >  EPS and iter < max_iter):
        prev_t = t.copy()
        t -= alpha * df(x, y, t)
        if iter % 2000 == 0:
            
            logger.debug("Iteration {}".format(iter))
            if not args.part == 7:
                logger.debug("x = ({:.2f}, {:.2f}, {:.2f}, ...,) f(x)={:.2f})".format(
                    t[0], t[1], t[2], f(x, y, t)))
            else:
                logger.debug("f(x)={:.2f}".format(f(x, y, t)))
            logger.debug("Gradient: {} \n".format(df(x, y, t)))
        iter += 1
    return (t,f(x, y, t))        
    
def f(x, y, theta):
    """cost function"""
    x = np.vstack((np.ones((1, x.shape[1])), x))
    return np.sum((y - np.dot(theta.T, x)) ** 2)

def df(x, y, theta):
    """derivative of cost function used in gradient descent"""
    x = np.vstack((np.ones((1, x.shape[1])), x))
    return -2 * np.sum((y - np.dot(theta.T, x)) * x, 1) 
    
def fv(x, y, theta):
    """cost function for one-hot encoding
       minimize the squared error of each label dimension
       theta is N by K
    """
    x = np.vstack((np.ones((1, x.shape[1])), x))
    return np.sum(np.sum((y - np.dot(theta.T, x)) ** 2, 0))
    
def dfv(x, y, theta):
    """derivation of cost function for one-hot encoding"""
    x = np.vstack((np.ones((1, x.shape[1])), x))
    return -2 * np.dot(x,(y - np.dot(theta.T, x)).T)

def h(x, opt_theta):
    """dot product of x, optimal theta"""
    return np.dot(x, opt_theta)
    
def hv(x, opt_theta):
    """one-hot encoding version of h"""
    return np.dot(opt_theta.T, x)

def accuracy(actor_list, dataset, opt_theta, size=10, female_list=[], male_list=[]):
    """test accuracy of theta on the validation and test set"""
    
    # for actor classification (part 3)
    actor_score = {actor: 0 for actor in actor_list}
    
    # for gender classification (part 5)
    male_score = 0
    female_score = 0
    
    # for one-hot classification (part 7)
    score = 0
    
    for actor in actor_list:
        if not os.path.exists(os.path.join('dataset', dataset)):
            raise ValueError("The directory {} does not exist under dataset/".format(dataset))
            
        start, end = get_range(actor, os.path.join('dataset', dataset))
        pic_names = os.listdir(os.path.join('dataset', dataset))
        
        for index in range(start, end + 1):
            if (index - start >= size):
                break
                
            # True to keep image as grey scale, scipy opens to RGB by default
            image = imread(os.path.join(os.path.join('dataset', dataset), pic_names[index]), True)
            
            # flatten to column vector (1024L,)
            im = np.reshape(image, (1024))
            
            # add the bias
            im = np.hstack((1, im))
            
            # if part 3, do actor classification
            if args.part == 3:
                h_ = h(im, opt_theta)
                if h_ >= 0.5 and (actor == 'hader'):
                    actor_score['hader'] += 1
                elif h_ < 0.5 and (actor == 'carell'):
                    actor_score['carell'] += 1
                    
            # if part 5, do gender classification
            elif args.part == 5:
                h_ = h(im, opt_theta)
                if h_ >= 0.5 and (actor in female_list):
                    female_score += 1
                elif h_ < 0.5 and (actor in male_list):
                    male_score += 1
                    
            # if part 7, do one-hot encoding classification
            elif args.part == 7:
                h_ = hv(im, opt_theta)
                max_index = np.argmax(h_)
                
                if (max_index == 0) and actor == "drescher":
                    score += 1
                elif (max_index == 1) and actor == "ferrera":
                    score += 1
                elif (max_index == 2) and actor ==  "chenoweth":
                    score += 1
                elif (max_index == 3) and actor ==  "baldwin":
                    score += 1
                elif (max_index == 4) and actor == "hader":
                    score += 1
                elif (max_index == 5) and actor ==  "carell":
                    score += 1
                
    if args.part == 3:            
        return actor_score
    elif args.part == 5:
        return float(male_score + female_score) / (size * len(actor_list))
    elif args.part == 7:
        return float(score) / (size * len(actor_list))
        
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
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

