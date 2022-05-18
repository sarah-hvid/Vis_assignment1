"""
Script that uses feature extraction and nearest neighbours to find similar images. The user may input a single image or a directory of images (png or jpg). 
"""
# base tools
import os, sys
import glob
import argparse
import pandas as pd

# data analysis
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

# tensorflow
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# function that specifies the required arguments
def parse_args():
    # Initialise argparse
    ap = argparse.ArgumentParser()
    
    # command line parameters
    ap.add_argument("-f", "--file_input", required = True, help = "The filename or directory you want to work with")
        
    args = vars(ap.parse_args())
    return args


# function to load vgg16
def load_vgg16():
    model = VGG16(weights = 'imagenet', # default parameter
              include_top = False, # remove classifier
              pooling = 'avg',
              input_shape = (224, 224, 3)) # define new images input shape
    return model


# function to extract features from image data using pretrained model (e.g. VGG16)
def extract_features(img_path, model):
    
    # Define input image shape
    input_shape = (224, 224, 3)
    
    # load image from file path
    img = load_img(img_path, target_size=(input_shape[0], 
                                          input_shape[1]))
    img_array = img_to_array(img)
    
    # expand to fit dimensions
    expanded_img_array = np.expand_dims(img_array, axis=0)
    
    # preprocess image
    preprocessed_img = preprocess_input(expanded_img_array)
    
    # use the predict function to create feature representation
    features = model.predict(preprocessed_img)
    
    # flatten and normalise features
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(features)

    return normalized_features


# function to clean a file path name
def clean_name(name):
    split_filename = name.split("/")[-1]
    file_only = split_filename.split(".")[0]
    
    return file_only


# function to extract features from all images in the input directory
def list_features(model):
    
    # parse argument
    args = parse_args()
    input_name = args['file_input']
    
     # if the path provided is a file
    isFile = os.path.isfile(input_name)
    if isFile == True:
        input_name = input_name.split("/")[:-1]
        input_name = '/'.join(input_name) # remove the filename to get the directory
    
    # get all full path names
    joined_paths = glob.glob(os.path.join(input_name, '*g'))
    joined_paths = sorted(joined_paths)
    
    feature_list = []
    
    # for every image file in the directory
    for file in sorted(joined_paths):
        features = extract_features(file, model)
        feature_list.append(features)
        
    return joined_paths, feature_list


# function to plot target image with the three nearest neighbours
def plot_nn(input_idx, joined_paths, dist, idxs):
    
     # create plot layout
    fig, ax = plt.subplots(1,4, figsize=(10, 5))
    # set titles 
    fig.suptitle('Most similar images - nearest neighbors', fontsize = 25)
    ax[0].set_title('Main image')
    ax[1].set_title(dist[0])
    ax[2].set_title(dist[1])
    ax[3].set_title(dist[2])

    # plot all images
    ax[0].imshow(mpimg.imread(joined_paths[input_idx]), extent=[-4, 4, -1, 1], aspect=4) # aligning the size of the images
    ax[1].imshow(mpimg.imread(joined_paths[idxs[0]]), extent=[-4, 4, -1, 1], aspect=4)
    ax[2].imshow(mpimg.imread(joined_paths[idxs[1]]), extent=[-4, 4, -1, 1], aspect=4)
    ax[3].imshow(mpimg.imread(joined_paths[idxs[2]]), extent=[-4, 4, -1, 1], aspect=4)

    # remove ticks from plots
    for i in ax:
        i.set_xticks([])
        i.set_yticks([])
        
    return 


# function to calculate the nearest neighbours for the target image
def calculate_nn(joined_paths, feature_list, input_name):
    
    filename = clean_name(input_name)
    
    output_data = []
    output_data.append(filename)
    
    # get the index of the input file name
    for idx, path in enumerate(joined_paths):
        if path == input_name:
            input_idx = idx
    
    
    # calculate nearest neighbors
    neighbors = NearestNeighbors(n_neighbors = 10,
                             algorithm = 'brute',
                             metric = 'cosine').fit(feature_list)
    
    # get nn for target
    distances, indices = neighbors.kneighbors([feature_list[input_idx]])
    
    # iterate to get distance and index of the three nearest
    dist = []
    idxs = []
    
    for i in range(1,4):
        idxs.append(indices[0][i])
        dist.append(round(distances[0][i], 3))
    
    for i in range(0,3):
        # saving the results of each image in the output list
        image_name = joined_paths[idxs[i]]
        image_name = clean_name(image_name)
        
        output_data.append(image_name) 
        output_data.append(dist[i])
        
    return dist, idxs, output_data


# function to create plot and df output for one file
def nn_one_result(joined_paths, feature_list):
    
    # parse argument
    args = parse_args()
    input_name = args['file_input']
    filename = clean_name(input_name)
    
    # get the index of the input file name
    for idx, path in enumerate(joined_paths):
        if path == input_name:
            input_idx = idx
    
    # get nearest neighbours
    dist, idxs, output_data = calculate_nn(joined_paths, feature_list, input_name)
        
    # creating the dataframe and transposing it to get columns
    df = pd.DataFrame(output_data, index = ["main_image", "1_image", "1_image_score", "2_image", "2_image_score", "3_image", "3_image_score"])
    df = df.transpose()
    df.to_csv(f'output/nn_{filename}.csv', index = False)
    
    # create and save plot
    plot_nn(input_idx, joined_paths, dist, idxs)
    plt.savefig(f'output/nn_{filename}.jpg')
    
    return 


# function to create df output for all files
def nn_all_results(joined_paths, feature_list):

    out_list = []
    
    # calculate nearest neighbors for each file
    for file in joined_paths:
        dist, idxs, output_data = calculate_nn(joined_paths, feature_list, file)
      
        out_list.append(output_data) # appending the list of the output data to the out_list for the dataframe
        
    df = pd.DataFrame(out_list, columns = ["main_image", "1_image", "1_image_score", "2_image", "2_image_score", "3_image", "3_image_score"])
    df.to_csv(f'output/nn_all_files.csv', index = False)
    
    return


def main():
    
    # parse arguments
    args = parse_args()
    input_name = args['file_input']
   
    # if path provided is a file:
    isFile = os.path.isfile(input_name)
    if isFile == True:
        model = load_vgg16()
        joined_paths, feature_list = list_features(model)
        nn_one_result(joined_paths, feature_list)
        
        print('Input is a file. Script success.')

    # if path provided is a directory:
    isDirectory = os.path.isdir(input_name)
    if isDirectory == True:
        model = load_vgg16()
        joined_paths, feature_list = list_features(model)
        nn_all_results(joined_paths, feature_list)
        
        print('Input is a directory. Script success.')
        
    return


if __name__ == '__main__':
    main()