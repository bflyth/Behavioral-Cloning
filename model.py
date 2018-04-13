import csv
import cv2
from PIL import Image
from numpy import repeat, array, append
import matplotlib.image as mpimg
from matplotlib import pyplot as plt



DATA_CSV_PATH = r"driving_log.csv"
DATA_IMG_PATH = r"IMG"

OUTPUT_FILE = "model.h5"

CAMERA_ANGLE_CORRECTION = .33 #a value for steering angle to camera correction



def import_csv_file(csv_path):
    """load image references from csv file"""
    lines = []
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        [lines.append(line) for line in reader]
    return array(lines)

def process_images(csv_lines, img_path):
    """
    forms a list of images from center, left and right sections of csv file
    dumps into array @images
    """
    images=[]
    for line in csv_lines[0:]:
        center, left, right = line[0], line[1], line[2]
        cameras = [center, left, right]
        for camera in cameras:
            filename = camera.split('\\')[-1] #splits along slashes in path and selects last item, the actual image name
            path = img_path + '/' + filename #recombine
            images.append(cv2.imread(path))
    images = array(images)
    return images

def process_steering(csv_lines):
    """
    forms a list of steering angle values to be passed to model
    """
    measurements = []
    for line in csv_lines[0:]:
        center = line[3]
        left = float(line[3]) + CAMERA_ANGLE_CORRECTION
        right = float(line[3]) + -CAMERA_ANGLE_CORRECTION
        angles = [center, left, right]
        [measurements.append(angle) for angle in angles]
    return array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

def model_arch(X_train, y_train, outfile):
    
    shape_input = (160, 320, 3)
    
    model = Sequential()
    
    model.add(Lambda(lambda x: x / 63.75 - 2.0, input_shape=shape_input)) #normalize images
    
    model.add(Cropping2D(cropping=((80, 20), (0, 0))))  #remove top 80 and bottom 20 pixels from normal-sized image
                                                        #to be rid of car hood and trees
    
    model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    
    model.add(Dropout(0.2)) #spice up and reduce overfit
    
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 2, 2, activation='relu'))

    model.add(Flatten())
    
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
    
    model.fit(X_train, y_train, batch_size=128, validation_split=0.33, verbose=2, shuffle=True, nb_epoch=5)

    model.save(outfile)
    print("Model Saved to directory as model.h5")

if __name__ == '__main__':
    data = import_csv_file(DATA_CSV_PATH)
    X_train = process_images(data, DATA_IMG_PATH)
    y_train = process_steering(data)
    model_arch(X_train, y_train, OUTPUT_FILE)
