#Python Plays: Grand Theft Auto V

#Video 9 : Create Training Data 

### Step 1 : record the key pressed
#input : images 
#output : actions 
#conduct the vehicle and record the keys press 
#getkeys.py : need to download pywin32 (python extension)
#In create_training_data.py : from getkeys import key_check


### Step 2 : convert keys to a multi-hot array
# create def keys_to_output(keys): 
# def keys_to_output(keys):
    #[A,W,D] boolean values.
    #output = [0,0,0]
    #if 'A' in keys:
        #output[0] = 1
    #elif 'D' in keys:
        #output[2] = 1
    #else:
        #output[1] = 1
    #return output

### Step 3 : save the different records 
#file_name = 'training_data.npy'

#if os.path.isfile(file_name):
    #print('File exists, loading previous data!')
    #training_data = list(np.load(file_name))
#else:
    #print('File does not exist, starting fresh!')
    #training_data = []

### Step 4 : extract the training data 
# extract the image and resize it 
# extract the list of keys pressed
# in def main():
# screen = grab_screen(region=(0,40,800,640))
            #last_time = time.time()
            #screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            #screen = cv2.resize(screen, (160,120))
            # resize to something a bit more acceptable for a CNN
            #keys = key_check()
            #output = keys_to_output(keys)
            #training_data.append([screen,output])
            #...np.save(file_name,training_data)


import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os


def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array

    [A,W,D] boolean values.
    '''
    output = [0,0,0]
    
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output


file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []


def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)


    paused = False
    while(True):

        if not paused:
            # 800x600 windowed mode
            screen = grab_screen(region=(0,40,800,640))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160,120))
            # resize to something a bit more acceptable for a CNN
            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen,output])
            
            if len(training_data) % 1000 == 0:
                print(len(training_data))
                np.save(file_name,training_data)

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)

main()

------------------------------------------------------------

#Video 10 : Balancing Self Driving Data 
# balance_data.py
# print(df.head()) 
# result : first column [0] = image, second column [1] = keys pressed
# print(Counter(df[1].apply(str)))
# result : Counter ({'[0,1,0]: 70375',[0,0,1]: 6708',[1,0,0]: 6427})
# balance means all the data must have a lenght of 6427 
#forwards = forwards[:len(lefts)][:len(rights)]
#lefts = lefts[:len(forwards)]
#rights = rights[:len(forwards)]
#final_data = forwards + lefts + rights
#np.save('training_data.npy', final_data)

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

train_data = np.load('training_data.npy')

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

lefts = []
rights = []
forwards = []

shuffle(train_data)

for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1,0,0]:
        lefts.append([img,choice])
    elif choice == [0,1,0]:
        forwards.append([img,choice])
    elif choice == [0,0,1]:
        rights.append([img,choice])
    else:
        print('no matches')


forwards = forwards[:len(lefts)][:len(rights)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]

final_data = forwards + lefts + rights
shuffle(final_data)

np.save('training_data.npy', final_data)

------------------------------------------------------------------------------


#Video 11 : Implement Alexnet in the train_model.py file 
#Create alexnet.py 

### step 1 : initialize the parameters of the model : width, height, lr, epochs, name

### step 2 : create the model 
#import numpy as np
#from alexnet import alexnet
#model = alexnet(WIDTH, HEIGHT, LR)

### step 3 : create train and testing set 
#train = train_data[:-100]
		#X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
        #Y = [i[1] for i in train]
#test = train_data[-100:]
        #test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
        #test_y = [i[1] for i in test]

### step 4: apply the model to the training set and save the model 
#model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
#model.save(MODEL_NAME)


# train_model.py file 
import numpy as np
from alexnet import alexnet
WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

hm_data = 22
for i in range(EPOCHS):
    for i in range(1,hm_data+1):
        train_data = np.load('training_data-{}-balanced.npy'.format(i))

        train = train_data[:-100]
        test = train_data[-100:]

        X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}), 
            snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

        model.save(MODEL_NAME)
-----------------------------------------------------------------------------------------



