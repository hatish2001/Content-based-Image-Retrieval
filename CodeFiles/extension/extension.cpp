/*

Authored by: Aadhi Aadhavan Balasubramanian, Harishraj Udaya Bhaskar
Date: 01/26/2024

*/


#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

// Function to compute chi-squared distance between two histograms
double computeChiSquaredDistance(const Mat& hist1, const Mat& hist2) {
    double distance = 0.0;
    for (int i = 0; i < hist1.rows; ++i) {
        for (int j = 0; j < hist1.cols; ++j) {
            double a = hist1.at<float>(i, j);
            double b = hist2.at<float>(i, j);
            if (a + b != 0) {
                distance += pow(a - b, 2) / (a + b);
            }
        }
    }
    return distance;
}

// Function to compute multi-histogram distance
double computeMultiHistogramDistance(const Mat& hist1, const Mat& hist2) {
    return computeChiSquaredDistance(hist1, hist2);
}

// Function to compute RGB histogram for a given image
Mat computeRGChromaticityHistogram(const Mat& image, int numBins) {
    Mat hist;

    // Convert image to float
    Mat floatImage;
    image.convertTo(floatImage, CV_32F);

    // Compute RG chromaticity histogram
    Mat rgHist;
    int histSize[] = { numBins, numBins };
    float rRanges[] = { 0, 256 };
    float gRanges[] = { 0, 256 };
    const float* ranges[] = { rRanges, gRanges };
    int channels[] = { 0, 1 };
    calcHist(&floatImage, 1, channels, Mat(), rgHist, 2, histSize, ranges, true, false);

    // Normalize histogram
    normalize(rgHist, hist, 0, 1, NORM_MINMAX, -1, Mat());

    return hist;
}

int main() {
    // Open the camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open camera." << endl;
        return 1;
    }

    // Read images from the directory and compute their histograms
    string imageDirPath = "/Users/aadhi/Desktop/CS5330/Project2/olympus"; // Update this with your image directory path
    vector<pair<Mat, string>> images;
    for (const auto& entry : fs::directory_iterator(imageDirPath)) {
        string imagePath = entry.path().string();
        Mat image = imread(imagePath);
        if (image.empty()) {
            cerr << "Error: Unable to read image " << imagePath << endl;
            continue;
        }
        Mat hist = computeRGChromaticityHistogram(image, 16);
        images.emplace_back(hist, imagePath);
    }

    // Main loop
    while (true) {
        // Capture a frame from the camera
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Unable to capture frame from camera." << endl;
            break;
        }
        imshow("Camera Feed", frame);

        // Compute RGB histogram for the captured frame
        Mat frameHist = computeRGChromaticityHistogram(frame, 16);

        // Compute distances and find the closest image
        double minDistance = numeric_limits<double>::max();
        string closestImage;
        for (const auto& [hist, imagePath] : images) {
            double distance = computeMultiHistogramDistance(frameHist, hist);
            if (distance < minDistance) {
                minDistance = distance;
                closestImage = imagePath;
            }
        }

        // Display the closest image
        Mat closestImageMat = imread(closestImage);
        if (!closestImageMat.empty()) {
            imshow("Closest Image", closestImageMat);
        } else {
            cerr << "Error: Unable to read closest image." << endl;
        }

        // Refresh every 1 second
        if (waitKey(1000) == 27) {  // Escape key
            break;
        }
    }

    // Release the camera
    cap.release();
    destroyAllWindows();

    return 0;
}
