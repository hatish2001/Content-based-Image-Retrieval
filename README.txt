# Content-Based Image Retrieval (CBIR) System

## Project Description

This project is an implementation of a Content-Based Image Retrieval (CBIR) system in C++. The system is designed to develop a robust image matching algorithm by employing various feature extraction techniques and distance metrics. It incorporates several tasks, each focusing on different aspects of image matching and retrieval.

### Authors

- Harishraj Udaya Bhaskar
- Aadhi Aadhavan Balasubramanian 

## Features

The CBIR system comprises the following key components:

1. **Single Histogram Matching**: Utilizes RGB histograms and compares them using the sum-of-squared-difference as the distance metric.
2. **Histogram Matching**: Implements histogram matching with a normalized color histogram, using histogram intersection as the distance metric.
3. **Multi-Histogram Matching**: Enhances histogram matching by using multiple histograms representing different spatial parts of the image, combined through weighted averaging.
4. **Feature Vector Matching**: Employs pre-computed feature vectors from a CSV file and compares them using cosine distance as the distance metric.
5. **CBIR System Integration**: Integrates the feature vectors from the CSV file with histogram matching using chi-squared distance as the distance metric for comprehensive image retrieval.

The project leverages the OpenCV library for image processing tasks, Boost libraries for file system operations, and implements custom distance metrics and feature extraction techniques. A live demonstration of the image retrieval process using the on-device camera feed is included, continuously displaying the closest matching image from a database directory.

## Requirements

- C++ Compiler (C++11 or higher recommended)
- OpenCV Library
- Boost Libraries

## Installation and Setup

1. Clone this repository to your local machine.
2. Ensure you have OpenCV and Boost libraries installed on your system.
3. Compile the project using a C++ compiler. Example compilation command:
4. Run the compiled executable to start the CBIR system.

## Usage

To use the CBIR system, follow these steps:

1. Place your image dataset in the specified directory structure.
2. Run the compiled CBIR system executable.
3. The system will process the images and perform image retrieval based on the camera feed or an input image.

## Contributing

We welcome contributions to this project! If you have suggestions or improvements, please fork the repository and submit a pull request.


