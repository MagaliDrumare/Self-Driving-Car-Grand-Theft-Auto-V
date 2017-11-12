#How to improve the model (V 0.03)

# // Increase the processing power. 

# 1-improve/ test # CNN models googlenet

# 2-improve the collection of the data in collect_data.py
#increase the size of the image : screen = cv2.resize(screen, (480, 270)) ~ ability to read the mini-map. 
# increase the number of actions recorded. 
#w = [1,0,0,0,0,0,0,0,0]
#s = [0,1,0,0,0,0,0,0,0]
#a = [0,0,1,0,0,0,0,0,0]
#d = [0,0,0,1,0,0,0,0,0]
#wa = [0,0,0,0,1,0,0,0,0]
#wd = [0,0,0,0,0,1,0,0,0]
#sa = [0,0,0,0,0,0,1,0,0]
#sd = [0,0,0,0,0,0,0,1,0]
#nk = [0,0,0,0,0,0,0,0,1]

# 3-Increase the number of output in train_model.py 
#model = googlenet(WIDTH, HEIGHT, 3, LR, output=9, model_name=MODEL_NAME)


# 4-Increase the number of actions in the test_model.py 
 #prediction = np.array(prediction) * np.array([4.5, 0.1, 0.1, 0.1, 1.8, 1.8, 0.5, 0.5, 0.2])

            #mode_choice = np.argmax(prediction)

            #if mode_choice == 0:
                #straight()
                #choice_picked = 'straight'
            #elif mode_choice == 1:
                #reverse()
                #choice_picked = 'reverse'
            #elif mode_choice == 2:
                #left()
                #choice_picked = 'left'
            #elif mode_choice == 3:
                #right()
                #choice_picked = 'right'
            #elif mode_choice == 4:
                #forward_left()
                #choice_picked = 'forward+left'
            #elif mode_choice == 5:
                #forward_right()
                #choice_picked = 'forward+right'
            #elif mode_choice == 6:
                #reverse_left()
                #choice_picked = 'reverse+left'
            #elif mode_choice == 7:
                #reverse_right()
                #choice_picked = 'reverse+right'
            #elif mode_choice == 8:
                #no_keys()
                #choice_picked = 'nokeys'



#collect_data.py 
import os, time, cv2
import numpy as np
from grabscreen import grab_screen
from getkeys import key_check


key_map = {
    'W': [1, 0, 0, 0, 0, 0, 0, 0, 0],
    'S': [0, 1, 0, 0, 0, 0, 0, 0, 0],
    'A': [0, 0, 1, 0, 0, 0, 0, 0, 0],
    'D': [0, 0, 0, 1, 0, 0, 0, 0, 0],
    'WS': [0, 0, 0, 0, 1, 0, 0, 0, 0],
    'WD': [0, 0, 0, 0, 0, 1, 0, 0, 0],
    'SA': [0, 0, 0, 0, 0, 0, 1, 0, 0],
    'SD': [0, 0, 0, 0, 0, 0, 0, 1, 0],
    'NK': [0, 0, 0, 0, 0, 0, 0, 0, 1],
    'default': [0, 0, 0, 0, 0, 0, 0, 0, 0],
}

starting_value = 1058

while True:
    file_name = 'training_data-{0}.npy'.format(starting_value)
    if os.path.isfile(file_name):
        print('File exists, moving along', starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!', starting_value)
        break

def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    if ''.join(keys) in key_map:
        return key_map[''.join(keys)]
    return key_map['default']


def main(file_name, starting_value):
    training_data = []
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    last_time = time.time()
    paused = False
    print('STARTING!!!')
    while(True):
        if not paused:
            screen = grab_screen(region = (0, 40, 1920, 1120))
            last_time = time.time()
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (480, 270))
            # run a color convert:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen,output])

##            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
##            cv2.imshow('window',cv2.resize(screen,(640,360)))
##            if cv2.waitKey(25) & 0xFF == ord('q'):
##                cv2.destroyAllWindows()
##                break

            if len(training_data) % 100 == 0:
                print(len(training_data))
                if len(training_data) == 500:
                    np.save(file_name, training_data)
                    print('SAVED')
                    training_data = []
                    starting_value += 1
                    file_name = 'X:/pygta5/phase7-larger-color/training_data-{0}.npy'.format(starting_value)

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('Unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)

if __name__ == "__main__":
    main(file_name, starting_value)


# train_model.py
import os, cv2
import pandas as pd
import numpy as np
from grabscreen import grab_screen
from tqdm import tqdm
from collections import deque
from models import inception_v3 as googlenet
from random import shuffle


FILE_I_END = 1860

WIDTH = 480
HEIGHT = 270
LR = 1e-3
EPOCHS = 30

MODEL_NAME = ''
PREV_MODEL = ''
LOAD_MODEL = True

model = googlenet(WIDTH, HEIGHT, 3, LR, output=9, model_name=MODEL_NAME)

if LOAD_MODEL:
    model.load(PREV_MODEL)
    print('We have loaded a previous model!!!!')

# iterates through the training files
for e in range(EPOCHS):
##    data_order = [i for i in range(1,FILE_I_END+1)]
    data_order = [i for i in range(1, FILE_I_END + 1)]
    shuffle(data_order)
    for count, i in enumerate(data_order):
        try:
            file_name = 'J:/phase10-random-padded/training_data-{0}.npy'.format(i)
            # full file info
            train_data = np.load(file_name)
            print('training_data-{0}.npy'.format(i), len(train_data))

##            # [   [    [FRAMES], CHOICE   ]    ]
##            train_data = []
##            current_frames = deque(maxlen=HM_FRAMES)
##
##            for ds in data:
##                screen, choice = ds
##                gray_screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
##
##
##                current_frames.append(gray_screen)
##                if len(current_frames) == HM_FRAMES:
##                    train_data.append([list(current_frames),choice])

            # always validating unique data:
##            shuffle(train_data)
            train = train_data[:-50]
            test = train_data[-50:]

            X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 3)
            Y = [i[1] for i in train]

            test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 3)
            test_y = [i[1] for i in test]

            model.fit({'input': X}, {'targets': Y}, n_epoch = 1, validation_set = ({'input': test_x}, {'targets': test_y}),
                snapshot_step = 2500, show_metric = True, run_id = MODEL_NAME)

            if count % 10 == 0:
                print('SAVING MODEL!')
                model.save(MODEL_NAME)

        except Exception as e:
            print(e)


#tensorboard --logdir=foo:J:/phase10-code/log


# test_model.py 
import cv2, time, random
import numpy as np
from grabscreen import grab_screen
from getkeys import key_check
from collections import deque, Counter
from models import inception_v3 as googlenet
from directkeys import PressKey, ReleaseKey, W, A, S, D
from statistics import mode, mean
from motion import motion_detection


GAME_WIDTH = 1920
GAME_HEIGHT = 1080

how_far_remove = 800
rs = (20, 15)
log_len = 25

motion_req = 800
motion_log = deque(maxlen = log_len)

WIDTH = 480
HEIGHT = 270
LR = 1e-3
EPOCHS = 10

choices = deque([], maxlen = 5)
hl_hist = 250
choice_hist = deque([], maxlen = hl_hist)

t_time = 0.25

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def left():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
##    ReleaseKey(S)


def right():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)


def no_keys():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


model = googlenet(WIDTH, HEIGHT, 3, LR, output = 9)
MODEL_NAME = ''
model.load(MODEL_NAME)

print('We have loaded a previous model!!!!')

def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    mode_choice = 0

    screen = grab_screen(region = (0, 40, GAME_WIDTH, GAME_HEIGHT + 40))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    prev = cv2.resize(screen, (WIDTH,HEIGHT))

    t_minus, t_now, t_plus = prev, prev, prev

    while(True):
        if not paused:
            screen = grab_screen(region = (0, 40, GAME_WIDTH, GAME_HEIGHT + 40))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            last_time = time.time()
            screen = cv2.resize(screen, (WIDTH,HEIGHT))

            delta_count_last = motion_detection(t_minus, t_now, t_plus)

            t_minus = t_now
            t_now = t_plus
            t_plus = screen
            t_plus = cv2.blur(t_plus, (4, 4))

            prediction = model.predict([screen.reshape(WIDTH,HEIGHT, 3)])[0]
            prediction = np.array(prediction) * np.array([4.5, 0.1, 0.1, 0.1, 1.8, 1.8, 0.5, 0.5, 0.2])

            mode_choice = np.argmax(prediction)

            if mode_choice == 0:
                straight()
                choice_picked = 'straight'
            elif mode_choice == 1:
                reverse()
                choice_picked = 'reverse'
            elif mode_choice == 2:
                left()
                choice_picked = 'left'
            elif mode_choice == 3:
                right()
                choice_picked = 'right'
            elif mode_choice == 4:
                forward_left()
                choice_picked = 'forward+left'
            elif mode_choice == 5:
                forward_right()
                choice_picked = 'forward+right'
            elif mode_choice == 6:
                reverse_left()
                choice_picked = 'reverse+left'
            elif mode_choice == 7:
                reverse_right()
                choice_picked = 'reverse+right'
            elif mode_choice == 8:
                no_keys()
                choice_picked = 'nokeys'

            motion_log.append(delta_count)
            motion_avg = round(mean(motion_log), 3)
            print('loop took {0} seconds. Motion: {1}. Choice: {2}'.format(round(time.time() - last_time, 3), motion_avg, choice_picked))

            if motion_avg < motion_req and len(motion_log) >= log_len:
                print('WERE PROBABLY STUCK FFS, initiating some evasive maneuvers.')

                # 0 = reverse straight, turn left out
                # 1 = reverse straight, turn right out
                # 2 = reverse left, turn right out
                # 3 = reverse right, turn left out

                quick_choice = random.randrange(0, 4)

                if quick_choice == 0:
                    reverse()
                    time.sleep(random.uniform(1, 2))
                    forward_left()
                    time.sleep(random.uniform(1, 2))

                elif quick_choice == 1:
                    reverse()
                    time.sleep(random.uniform(1, 2))
                    forward_right()
                    time.sleep(random.uniform(1, 2))

                elif quick_choice == 2:
                    reverse_left()
                    time.sleep(random.uniform(1, 2))
                    forward_right()
                    time.sleep(random.uniform(1, 2))

                elif quick_choice == 3:
                    reverse_right()
                    time.sleep(random.uniform(1, 2))
                    forward_left()
                    time.sleep(random.uniform(1, 2))

                for i in range(log_len - 2):
                    del motion_log[0]

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

if __name__ == "__main__":
    main()

 