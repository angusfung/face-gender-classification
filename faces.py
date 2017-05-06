from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
#from scipy.misc import imsave
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import pandas as pd
from numpy import *
from numpy.linalg import norm

## IMPORTANT: READ APPENDIX BEFORE RUNNING ##


##GET DATA
    
##RUN THIS TO DOWNLOAD THE PICTURES##
def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()            


#Note: you need to create the uncropped folder first in order 
#for this to work

def rgb2gray(rgb):
    #Return the grayscale version of the RGB image rgb as a 2D numpy array
    #whose range is 0..1
    #Arguments:
    #rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    #range of the values is 0..255
    
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.
    

    
def get_crop_pictures(filename,act):
    
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open(filename):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                
                #uncropped1= picture sourced from website -> effectively "uncropped"
                #cropped2 = the actual cropped image -> effectively "cropped"
                
                #don't increment if no picture was retrieved (?)
                if not os.path.isfile("uncropped/"+filename):
                    continue
    
                
                print filename
                i += 1
                
                #retrieve bounded box coordinates
                x1 = int(line.split()[5].split(",")[0])
                y1 = int(line.split()[5].split(",")[1])
                x2 = int(line.split()[5].split(",")[2])
                y2 = int(line.split()[5].split(",")[3])
            
    
                try: 
                    im = imread("uncropped/" + filename)
                except: #picture is corrupted, need to overwrite the picture.
                        #decrement the counter i.
                    i -= 1
                    continue
                
                #some pictures are already grey-scale, or are already 2D 
                
                try: 
                    im = im[y1:y2, x1:x2, :] #3D
                except:
                    im = im[y1:y2, x1:x2] #2D, don't need grey-scale
                    
                    #saving a blank picture that has no pixels, cannot be resized.
                    try:
                        im = imresize(im, (32,32)) #add 3
                        imsave("cropped/" + filename, im)
                    except:
                            #picture is blank, need to overwrite the picture.
                            #decrement the counter i.
                            i -= 1
                            continue
                    continue
                
                #saving a blank picture that has no pixels, cannot be resized.
                try:
                    im = rgb2gray(im)
                    im = imresize(im, (32,32)) #add 3
                    imsave("cropped/" + filename, im)
                    
                except:
                        #picture is blank, need to overwrite the picture.
                        #decrement the counter i.
                        i -= 1
                        continue
    return
    
def download_pictures():
    act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

    #act = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan', 'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']

    os.makedirs("uncropped")
    os.makedirs("cropped")
    get_crop_pictures("facescrub_actors.txt",act)
    get_crop_pictures("facescrub_actresses.txt",act)
    return
######################################################################
download = True #SET TO FALSE IF YOU DO NOT WANT TO DOWNLOAD PICTURES

if download:
    download_pictures()

##Part 2 (To make the training, validation and testing sets)

act =['drescher', 'ferrera', 'chenoweth', 'baldwin', 'hader', 'carell']


def makeset(act):
    #input: cropped folder, containing 1942 pictures of every actor.
    os.makedirs("trainingset")
    os.makedirs("validationset")
    os.makedirs("testset")
    random.seed(0)
    usedlist = []
    cropped = os.listdir("cropped")
    act =['drescher', 'ferrera', 'chenoweth', 'baldwin', 'hader', 'carell']
    
    for a in act:
        i=0
        while (i<100):
            index = int(random.random()*1941) #images from 0 to 1941 
            if a in cropped[index] and index not in usedlist:
                im = imread("cropped/"+cropped[index])
                imsave("trainingset/" + cropped[index], im)
                usedlist.append(index)
                i+=1
        
        i=0
        while (i<10):
            index = int(random.random()*1941) #images from 0 to 1941 
            if a in cropped[index] and index not in usedlist:
                im = imread("cropped/"+cropped[index])
                imsave("testset/" + cropped[index], im)
                usedlist.append(index)
                i+=1
        i=0
        while (i<10):
            index = int(random.random()*1941) #images from 0 to 1941 
            if a in cropped[index] and index not in usedlist:
                im = imread("cropped/"+cropped[index])
                imsave("validationset/" + cropped[index], im)
                usedlist.append(index)
                i+=1
        
    return        
                   

##Part 3 (Generating x and y)

act = ["hader", "carell"]

x = []
y = []
#create the M matrix, where M^T t = h
#create the y vector, with "hader=1", "carell=0"
for a in act:
    for picture in os.listdir("trainingset"):
        if a in picture:
            #print(picture)
            im = imread("trainingset/"+picture, True)
            im = reshape(im, (1024,1)) #1024 by 1 column vector 
            if (size(x)==0):
                x = im
            else:
                x = hstack((x,im))
            if (a == "hader") & (size(y)==0):
                y = 1
            elif a == "hader":
                y = vstack((y,1))
            elif size(y)==0:
                y = 0
            else:
                y=vstack((y,0)) 
y=reshape(y,(200))                 

##f and f'(x)


def f(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( (y - dot(theta.T,x)) ** 2)

def df(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return -2*sum((y-dot(theta.T, x))*x, 1)
           
##Gradient Descent (Michael Guerzhoy's Implementation)

def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-10   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 60000
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        if iter % 2000 == 0:
        #if iter % 1 == 0:
            print "Iter", iter
            print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
            print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return (t,f(x, y, t))
    
##h(x) = theta^T x

#initial conditions that generate the optimal theta.
#theta0=random.rand(1025)     
#theta0 = array([-0.04]*1025)
#theta = grad_descent(f, df, x, y, theta0, 0.0000000001)

            
#constructing h, x = 1024x1 size pixel, theta = 1025x1 coefficient


def h(x, theta):
    sum=0
    x = hstack((1,x))
    for i in range(len(theta)):
        sum += x[i]*theta[i]
    return sum
    
##Calculates accuracy of validation set and test set

act = ["hader", "carell"]
def testset(filename):
    hader = 0
    carell = 0
    for a in act:
        for picture in os.listdir(filename):
            if a in picture:
                
                im = imread(filename + "/" +picture, True)
                im = reshape(im, (1024))
                sum=h(im,theta[0])
                print(sum, picture)
                if (sum>0.5) & ("hader" in picture):
                    hader +=1
                elif (sum<0.5) & ("carell" in picture):
                    carell +=1
    return (hader, carell)
       
## Part 4a
       
#im=reshape(theta[0][1:], (32,32)) #theta[0] are the optimized theta, theta[1] is f_min
                                  #we only want theta[0][1:] which are all the values of 
                                  #theta, less the bias term.
#imshow(im)
#show()

## Part 4b
        
#using two images of each actor
#folder "cropped1" contains two pictures of actor Hader and Carell)

act = ["hader", "carell"]

x1 = []
y1 = []
#create the M matrix, where M^T t = h
#create the y vector, with "hader=1", "carell=0"
for a in act:
    for picture in os.listdir("cropped1"):
        if a in picture:
            im = imread("cropped1/"+picture, True)
            im = reshape(im, (1024,1)) #1024 by 1 column vector 
            if (size(x1)==0):
                x1 = im
            else:
                x1 = hstack((x1,im))
            if (a == "hader") & (size(y1)==0):
                y1 = 1
            elif a == "hader":
                y1 = vstack((y1,1))
            elif size(y1)==0:
                y1 = 0
            else:
                y1=vstack((y1,0)) 
y1=reshape(y1,(4))        

#theta0 = array([0.0]*1025)
#theta = grad_descent(f, df, x1, y1, theta0, 0.0000000001)


## Part 5

act =['drescher', 'ferrera', 'chenoweth', 'baldwin', 'hader', 'carell']
female= ['drescher', 'ferrera', 'chenoweth'] #y=1
male = ['baldwin', 'hader', 'carell'] #y=0

#SET THE TRAINING SET SIZE HERE
size=100
#validation set is 10 of each picture such that the intersection is the null set.
#choosen from the end of the list. 
#file: validationset

x1 = []
y1 = []

for a in act:
    i=0
    for picture in os.listdir("trainingset"):
        if a in picture and i <= size-1:
            i += 1
            im = imread("trainingset/"+picture, True)
            #save the pictures in a new folder, training set which we will later check on
            #accuracy
            #imsave("trainingset1/" + picture, im)
            im = reshape(im, (1024,1)) #1024 by 1 column vector 
            #print(picture)
            
            if (len(x1)==0):
                x1 = im
            else:
                x1 = hstack((x1,im))
            if (a in female) & (len(y1)==0):
                y1.extend([1])
            elif a in female:
                y1 = vstack((y1,1))
            elif len(y1)==0:
                y1.extend([0])
            else:
                y1=vstack((y1,0)) 
y1=reshape(y1,(size*6))                  

#gradient descent
#theta0=random.rand(1025)     
#theta0 = array([0.0]*1025)
#theta = grad_descent(f, df, x1, y1, theta0, 0.00000000001)

#test accuracy of validation set


def validationset(filename):
    female_num = 0
    male_num = 0
    for a in act:
        for picture in os.listdir(filename):
            if a in picture:
                
                im = imread(filename + "/" +picture, True)
                im = reshape(im, (1024))
                sum=h(im,theta[0])
                print(sum, picture)
                if (sum>0.5) and (a in female):
                    female_num +=1
                elif (sum<0.5) and (a in male):
                    male_num +=1
    return (female_num, male_num)

#test accuracy of training set 
def trainingset(filename):
    female_num = 0
    male_num = 0
    for a in act:
        i=0
        for picture in os.listdir(filename):
            if a in picture and i <= size-1:
                i += 1
                im = imread(filename + "/" +picture, True)
                im = reshape(im, (1024))
                sum=h(im,theta[0])
                print(sum, picture)
                if (sum>0.5) and (a in female):
                    female_num +=1
                elif (sum<0.5) and (a in male):
                    male_num +=1
    return (female_num, male_num)   
## PLOT
# 
# validationset = [60,55,72,68,82,80,77,78,78,77]
# trainingset = [100,100,100,100,100,100,99,99,99,99]
# size = [10,20,30,40,50,60,70,80,90,100]
# plot(size, validationset,size, trainingset)
# xlim(0,100)
# ylim(0,120)
# title('Performance vs. Size')
# xlabel('Training Set Size')
# ylabel('Performance')
# legend()
# legend(('Validation Set', 'Training Set'))
 


## Part 5b - New Actors
act_test = ['butler', 'radcliffe', 'vartan', 'bracco', 'gilpin', 'harmon']
female1= ['bracco', 'gilpin', 'harmon'] #y=1
male1 = ['butler', 'radcliffe', 'vartan'] #y=0

def validationset1(filename):
    female1_num = 0
    male1_num = 0
    for a in act_test:
        for picture in os.listdir(filename):
            if a in picture:
                
                im = imread(filename + "/" +picture, True)
                im = reshape(im, (1024))
                sum=h(im,theta[0])
                print(sum, picture)
                if (sum>0.5) and (a in female1):
                    female1_num +=1
                elif (sum<0.5) and (a in male1):
                    male1_num +=1
    return (female1_num, male1_num)
    
## Part 6 
#inputs:
#theta ~ (n+1)x k
#x ~ nxm
def fvector(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return sum(sum( (y - dot(theta.T,x)) ** 2,0))

def dfvector(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    #return -2*sum((y-dot(theta.T, x))*x, 1)
    return -2*dot(x,(y-dot(theta.T, x)).T)
## Verification of Derivative.
#checking that it works.
#assume 4 labels, so theta0 = n x 4 = 1025 x 4
# y = k x m = 4 x 200
'''uncomment to use.
random.seed(0)
y = reshape(random.rand(800), (4,200))
h = 0.000000001
theta0 = reshape(random.rand(4100),(1025,4))
dtheta = zeros((1025,4))
dtheta[0,0]=h #partial derivative with respect to element(1,1)
dtheta[0,1]=h #partial derivative with respect to the element (1,2)
dtheta[1,1]=h #partial derivative with respect to the element (2,2)
print (fvector(x, y, theta0+dtheta) - fvector(x, y, theta0-dtheta))/(2*h)
print dfvector(x, y, theta0)
'''

##Vectorized Gradient Descent

##Gradient Descent (Michael Guerzhoy's Implementation)

def grad_descentvector(f, df, x, y, init_t, alpha):
    EPS = 1e-10 #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter =50000
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*dfvector(x, y, t)
        if iter % 2000 == 0:
        #if iter % 1 == 0:
            print "Iter", iter
            print fvector(x, y, t)
            print "Gradient: ", dfvector(x, y, t), "\n"
        iter += 1
    return (t,fvector(x, y, t))
    
## Part 7

act =['drescher', 'ferrera', 'chenoweth', 'baldwin', 'hader', 'carell']

x1 = []
y1 = []

#construct the X matrix.
for a in act:
    for picture in os.listdir("trainingset"):
        if a in picture:
            im = imread("trainingset/"+picture, True)
            #save the pictures in a new folder, training set which we will later check on
            #accuracy
            #imsave("trainingset1/" + picture, im)
            im = reshape(im, (1024,1)) #1024 by 1 column vector 
            
            if (len(x1)==0):
                x1 = im
            else:
                x1 = hstack((x1,im))
                
            if len(y1)==0:
                if 'drescher' in a:
                    y1=reshape([1,0,0,0,0,0],(6,1))
                elif 'ferrera' in a:
                    y1=reshape([0,1,0,0,0,0],(6,1))
                elif 'chenoweth' in a:
                    y1=reshape([0,0,1,0,0,0],(6,1))
                elif 'baldwin' in a:
                    y1=reshape([0,0,0,1,0,0],(6,1))
                elif 'hader' in a:
                    y1=reshape([0,0,0,0,1,0],(6,1))
                elif 'carell'in a:
                    y1=reshape([0,0,0,0,0,1],(6,1))
            else: #len(y1)!
                if 'drescher' in a:
                    y1=hstack((y1,reshape([1,0,0,0,0,0],(6,1))))
                elif 'ferrera' in a:
                    y1=hstack((y1,reshape([0,1,0,0,0,0],(6,1))))
                elif 'chenoweth' in a:
                    y1=hstack((y1,reshape([0,0,1,0,0,0],(6,1))))
                elif 'baldwin' in a:
                    y1=hstack((y1,reshape([0,0,0,1,0,0],(6,1))))
                elif 'hader' in a:
                    y1=hstack((y1,reshape([0,0,0,0,1,0],(6,1))))
                elif 'carell' in a:
                    y1=hstack((y1,reshape([0,0,0,0,0,1],(6,1))))
#theta0=reshape(random.rand(1025*6), (1025,6))*0.01 #bound between 0 and 0.5   
theta0 = reshape([0.0]*1025*6, (1025,6))
theta = grad_descentvector(fvector, dfvector, x1, y1, theta0, 0.000000000001)                   


#evaluate h=theta.T X
def h1(x,theta):
    #arguments:
    #x 1025 x 1
    #theta 1025 x 6
    x = hstack((1,x))
    return dot(theta.T,x)
            
            
#test accuracy of training set and validation set
def accuracy(filename):
    total=0
    score=0
    for a in act:
        for picture in os.listdir(filename):
            if a in picture:
                total += 1
                im = imread(filename + "/" +picture, True)
                im = reshape(im, (1024))
                param=h1(im,theta[0]) #calculates the one-hot encoding
                max_index = argmax(param) #finds the index of the max
                print(max_index, picture)
                if (max_index == 0) and a in 'drescher':
                    score +=1
                elif (max_index == 1) and a in 'ferrera':
                    score +=1
                elif (max_index == 2) and a in 'chenoweth':
                    score +=1
                elif (max_index == 3) and a in 'baldwin':
                    score +=1
                elif (max_index == 4) and a in 'hader':
                    score +=1
                elif (max_index == 5) and a in 'carell':
                    score +=1
    return (score/float(total)) 
    

        
    
##Part 8 Visualization



im1=reshape(theta[0][1:,0], (32,32))#theta[0] are the optimized theta, theta[1] is f_min 
                                    #we only want theta[0][0,1:] which are all the values of 
                                    #theta, less the bias term.
im2=reshape(theta[0][1:,1], (32,32))
im3=reshape(theta[0][1:,2], (32,32))
im4=reshape(theta[0][1:,3], (32,32))
im5=reshape(theta[0][1:,4], (32,32))
im6=reshape(theta[0][1:,5], (32,32))
#imshow(im1)
#show()