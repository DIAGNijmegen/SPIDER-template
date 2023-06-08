#SPIDER Lumbar Spine Segmentation Challenge
Welcome to the SPIDER Lumbar Spine Segmentation Challenge Template Repository! This repository provides a template to help upload your lumbar spine segmentation algorithm to the Grand-Challenge platform. Follow the instructions below to customize and use this template effectively.

##Getting Started
To get started with the challenge, please follow the instructions below:

- Clone the Repository
Clone this repository to your local machine: https://github.com/DIAGNijmegen/SPIDER-template

- Install Dependencies
Make sure you have the required dependencies installed. This includes the TIGER library. 
To install the latest version, open a command prompt an run: `pip install git+https://github.com/DIAGNijmegen/msk-tiger.git@stable`

## Implementing your code

- Prepare Your Algorithm
Inside the src directory, you will find the SegmentVertebrae.py file. 
This is where you need to implement your lumbar spine segmentation algorithm. 
Replace the placeholder code (line 128) with your own implementation.

- Output variable
The two output variables needed to save the prediction are:
    - `total_segmentation`: A numpy array of similar shape as the input image which contains all masks of the segmented structures
    - `original_header`: The original header of the input image
 
- Build the docker with the provided dockerfile. 

