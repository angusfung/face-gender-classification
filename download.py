import urllib

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
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                                
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