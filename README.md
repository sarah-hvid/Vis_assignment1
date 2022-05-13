# Assignment 1 - Image search
 
Link to github of this assignment: https://github.com/sarah-hvid/Vis_assignment1

## Assignment description
In this assignment images that are alike should be found using their colour histograms. For a single main image, the 3 most similar images should be found. A plot and CSV file should be saved of the results.
The full assignment description is available in the ```assignment1.md``` file. 

## Methods
This problem relates to finding images that are alike.  The main image is compared to the rest of the images available in the ```data``` folder. The colours of the images are used to find similar images. A colour histogram is created for each image and the distance score between them is compared. The user may specify a single image or a directory containing multiple images. If one image is specified, a plot is created displaying the main image and the three most similar images along with their distance scores. A CSV file is also created with the same information. Both are saved in the ```output``` folder. If a directory is specified, only the CSV file is created. In this case all images are compared to all other images in the folder.

## Usage
In order to run the scripts, certain modules need to be installed. These can be found in the ```requirements.txt``` file. The folder structure must be the same as in this GitHub repository (ideally, clone the repository).
The data used in the assignment is the ```flowers``` dataset from the shared ```CDS-VIS``` folder. The data can also be downloaded from this website: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/. These images must be placed in the ```data``` folder in order to replicate the results. The current working directory when running the script must be the one that contains the ```data```, ```output``` and ```src``` folder.
Examples of how to run the scripts from the command line: 

Specifying a file:
- python src/hist_comparison.py -d data/image_0005.jpg
    
Specifying a directory:
- python src/hist_comparison.py -d data
  
Examples of the outputs of the scripts can be found in the ```output``` folder. 

## Results
The results of the colour histogram based image search is reasonable. It is clear that the colours are being used to find the images. Therefore, flowers that have the same colours are being marked as similar instead of flowers that are actually of the same species. The images themselves appear quite different from each other. This result could be improved using feature extraction of a pretrained model. 