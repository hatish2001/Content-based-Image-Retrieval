/*

Authored by: Aadhi Aadhavan Balasubramanian, Harishraj Udaya Bhaskar
Date: 01/26/2024

*/

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;


double computeChiSquaredDistance(const Mat& hist1, const Mat& hist2) {
    // Ensure the histograms have the same dimensions
    assert(hist1.size() == hist2.size());

    double distance = 0.0;
    for (int i = 0; i < hist1.rows; ++i) {
        for (int j = 0; j < hist1.cols; ++j) {
            double diff = hist1.at<float>(i, j) - hist2.at<float>(i, j);
            double sum = hist1.at<float>(i, j) + hist2.at<float>(i, j);
            if (sum != 0) {
                distance += (diff * diff) / sum;
            }
        }
    }

    return distance;
}

// Function to compute RG chromaticity histogram for a given image
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
    try {
        calcHist(&floatImage, 1, channels, Mat(), rgHist, 2, histSize, ranges, true, false);
    } catch (const cv::Exception& e) {
        cerr << "Error computing histogram: " << e.what() << endl;
        return Mat(); // Return empty histogram on error
    }

    // Normalize histogram
    normalize(rgHist, hist, 0, 1, NORM_MINMAX, -1, Mat());

    return hist;
}

// Function to compute correlation distance between two histograms
double computeCorrelationDistance(const Mat& hist1, const Mat& hist2) {
    return 1.0 - compareHist(hist1, hist2, HISTCMP_CORREL);
}




int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <target_image_path> <database_dir> <N>" << endl;
        return 1;
    }

    // Read command-line arguments
    string targetImagePath = argv[1];
    string databaseDir = argv[2];
    int N = stoi(argv[3]);

    // Read target image
    Mat targetImage = imread(targetImagePath);
    if (targetImage.empty()) {
        cerr << "Error: Unable to read target image." << endl;
        return 1;
    }

    // Compute histogram for the target image
    Mat targetHist = computeRGChromaticityHistogram(targetImage, 16);
    if (targetHist.empty()) {
        cerr << "Error: Unable to compute histogram for the target image." << endl;
        return 1;
    }

    // Loop over the directory of images
    vector<pair<double, string>> matches; // (distance, image_path) pairs
    for (const auto& entry : fs::directory_iterator(databaseDir)) {
        string imagePath = entry.path().string();
        Mat image = imread(imagePath);
        if (image.empty()) {
            cerr << "Error: Unable to read image " << imagePath << endl;
            continue;
        }

        // Compute histogram for the current image
        Mat imageHist = computeRGChromaticityHistogram(image, 16);
        if (imageHist.empty()) {
            cerr << "Error: Unable to compute histogram for image " << imagePath << endl;
            continue;
        }

        // Compute histogram intersection distance between target and current image
        double distance = computeChiSquaredDistance(targetHist, imageHist);

        // Store the result
        matches.push_back({distance, imagePath});
    }

    // Sort the list of matches based on distance
    sort(matches.begin(), matches.end());

    // Output the top N matches
    cout << "Top " << N << " matches:" << endl;
    for (int i = 0; i < min(N, (int)matches.size()); ++i) {
        cout << matches[i].second << " (Distance: " << matches[i].first << ")" << endl;
        
        // Display the top N closest images
        Mat closestImage = imread(matches[i].second);
        if (!closestImage.empty()) {
            imshow("Closest Image " + to_string(i+1), closestImage);
        }
    }

    // Display the target image
    imshow("Target Image", targetImage);
    waitKey(0);

    return 0;
}
