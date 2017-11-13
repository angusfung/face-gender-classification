import urllib
import os
import logging
from scipy.misc import imread
from scipy.misc import imresize
from pylab import imsave

logger = logging.getLogger(__name__)

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

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255'''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.
        
def get_crop_pictures(filename, act):
    
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open(filename):
            if a in line:
                filename = name + str(i) + '.' + line.split()[4].split('.')[-1]
                # timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "dataset/uncropped/" + filename), {}, 30)
                                
                # check if picture was copied
                if not os.path.isfile("dataset/uncropped/" + filename):
                    continue
            
                # check if image is valid
                try:
                    im = imread("dataset/uncropped/" + filename)
                    
                except IOError:
                    logger.info("Image from {} is corrupted".format(line.split()[4]))
                    continue
    
                logger.info("Copied {}".format(filename))   
                
                # retrieve bounded box coordinates
                x1 = int(line.split()[5].split(",")[0])
                y1 = int(line.split()[5].split(",")[1])
                x2 = int(line.split()[5].split(",")[2])
                y2 = int(line.split()[5].split(",")[3])
                 
                # check if pictures are already gray-scale (2D)
                
                try:
                    im = im[y1:y2, x1:x2, :]
                
                except IndexError:
                    logger.info("Image from {} is already gray-scaled".format(line.split()[4]))
                    im = im[y1:y2, x1:x2]
                    
                # check if picture is blank (has pixels)
                try:     
                    im = imresize(im, (32,32)) 
                except IOError:
                    logger.info("Image from {} is blank".format(line.split()[4]))
                    continue
                #imsave("dataset/cropped/" + filename, im)

                #     try:
                #         im = imresize(im, (32,32)) #add 3
                #         imsave("cropped/" + filename, im)
                #     except:
                #             #picture is blank, need to overwrite the picture.
                #             #decrement the counter i.
                #             i -= 1
                #             continue
                #     continue
                # 
                # #saving a blank picture that has no pixels, cannot be resized.
                # try:
                #     im = rgb2gray(im)
                #     im = imresize(im, (32,32)) #add 3
                #     imsave("cropped/" + filename, im)
                #     
                # except:
                #         #picture is blank, need to overwrite the picture.
                #         #decrement the counter i.
                #         i -= 1
                #         continue
                
                # increment
                i += 1
    return