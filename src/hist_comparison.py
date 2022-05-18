"""
A script that compares image histograms quantitively. The user must specify either a single image or a directory (jpg/png). 
"""
# system tools
import os
import argparse
import sys

# image and data tools
import cv2
import numpy as np
import glob
import pandas as pd

# plotting tools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# function that specifies the required arguments
def parse_args():
    # Initialise argparse
    ap = argparse.ArgumentParser()
    
    # command line parameters
    ap.add_argument("-f", "--file_input", required = True, help = "The filename or directory we want to work with")
        
    args = vars(ap.parse_args())
    return args


# function to clean a file path name
def clean_name(name):
    split_filename = name.split("/")[-1]
    file_only = split_filename.split(".")[0]
    
    return file_only


# function to create the histogram
def create_histogram(image):
     # creating histograms for image
    hist = cv2.calcHist([image], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256])
    hist = cv2.normalize(hist, hist, 0,255, cv2.NORM_MINMAX) #normalizing with MINMAX
    
    return hist


# function to calculate similarity scores
def calc_sim_scores(filename): 
    # parse argument
    args = parse_args()
    input_name = args['file_input']
    
     # if the path provided is a file
    isFile = os.path.isfile(input_name)
    if isFile == True:
        input_name = input_name.split("/")[:-1]
        input_name = '/'.join(input_name) # remove the filename to get the directory
    
    # get all full path names
    file_list = glob.glob(os.path.join(input_name, '*g'))
    file_list = sorted(file_list)
    
    target_image = mpimg.imread(filename) # loading target image
    target_hist = create_histogram(target_image)

    # initiating lists for results
    sim_value = [] 
    img_name = [] 

    for img in file_list: 
        if img != filename: # excluding the target image
            image = mpimg.imread(img)
            hist = create_histogram(image)

            score = round(cv2.compareHist(target_hist, hist, cv2.HISTCMP_CHISQR), 2) # comparing each histogram to the target image
            img_name.append(img) # appending results to the lists
            sim_value.append(score)

    # zipping the lists together, sorting it, and selecting the top 3 results.
    scores = sorted(zip(sim_value, img_name))
    scores = scores[0:3]
    
    return scores


# function to create the plot and dataframe of the results
def hist_result(file, scores):
    target_image = mpimg.imread(file) # loading target image
    file_name = clean_name(file)
    
    # initiating a list for the output dataframe
    output_data = [] 
    output_data.append(file_name)
    
    # initiating plot
    plt.subplot(1, 4, 1) #the figure has 1 row, 4 columns, and this is the first plot
    plt.imshow(target_image, extent=[-4, 4, -1, 1], aspect=4) # aligning the size of the images
    plt.axis('off')
    plt.title('Main image')

    plot_number = 1 # creating counter for the subplots

    for value, image in scores:
        value = round(value) # rounding the similarity score for a cleaner output
        image_name = clean_name(image)
        
        output_data.append(image_name) # saving the results of each image in the output list
        output_data.append(value)
        
        # plotting
        plot_number += 1
        image_sub = mpimg.imread(image)
        
        # creating the subplots
        plt.subplot(1, 4, plot_number)
        plt.imshow(image_sub, extent=[-4, 4, -1, 1], aspect=4)
        plt.axis('off')
        plt.title(f"Score: {value}")

    # Saving the plot
    plt.suptitle("Most similar images")
    plt.savefig(f'output/hist_{file_name}.jpg')
    
    # creating the dataframe and transposing it to get columns
    df = pd.DataFrame(output_data, index = ["main_image", "1_image", "1_image_score", "2_image", "2_image_score", "3_image", "3_image_score"])
    df = df.transpose()
   
    outpath = os.path.join('output', f'hist_{file_name}.csv')
    df.to_csv(outpath, index = False)
    
    return


# function to create a dataframe of the similarity scores for multiple input files
def hist_all_results(input_name): 
    
    # get all full path names
    file_list = glob.glob(os.path.join(input_name, '*g'))
    file_list = sorted(file_list)
    
    # initiate list for dataframe
    out_list = []
    for file in file_list:
        scores = calc_sim_scores(file)
        file_name = clean_name(file)
        
        output_data = [] # initiating output data for each main image
        output_data.append(file_name) 

        # cleaning the values and names, and appending them to the output data 
        for value, image in scores:
            value = round(value)
            image_name = clean_name(image)

            output_data.append(image_name)
            output_data.append(value)


        out_list.append(output_data) # appending the output data to the out_list for the dataframe

    # creating the dataframe
    df = pd.DataFrame(out_list, columns = ["main_image", "1_image", "1_image_score", "2_image", "2_image_score", "3_image", "3_image_score"])
    
    outpath = os.path.join('output', 'hist_all_files.csv')
    df.to_csv(outpath)
    
    return 
    

def main():
    
    # parse arguments
    args = parse_args()
    input_name = args['file_input']
   
    # if path provided is a file:
    isFile = os.path.isfile(input_name)
    if isFile == True:
        scores = calc_sim_scores(input_name)
        hist_result(input_name, scores)
        
        print('Input is a file. Script success.')

    # if path provided is a directory:
    isDirectory = os.path.isdir(input_name)
    if isDirectory == True:
        hist_all_results(input_name)
        
        print('Input is a directory. Script success.')
        
    return


if __name__ == '__main__':
    main()