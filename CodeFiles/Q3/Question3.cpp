/*

Authored by: Aadhi Aadhavan Balasubramanian, Harishraj Udaya Bhaskar
Date: 01/26/2024

*/

#include <iostream>
#include <vector>
#include <string>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

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


// Function to compute histogram intersection distance between two histograms
double computeChiSquareDistance(const Mat& hist1, const Mat& hist2) {
    // Ensure the histograms have the same dimensions
    assert(hist1.size() == hist2.size());

    double distance = 0.0;
    for (int i = 0; i < hist1.rows; ++i) {
        for (int j = 0; j < hist1.cols; ++j) {
            double numerator = pow(hist1.at<float>(i, j) - hist2.at<float>(i, j), 2);
            double denominator = hist1.at<float>(i, j) + hist2.at<float>(i, j);
            if (denominator != 0) {
                distance += numerator / denominator;
            }
        }
    }

    return distance;
}

// Function to compute multi-histogram distance
double computeMultiHistogramDistance(const Mat& hist1_top, const Mat& hist2_top, const Mat& hist1_bottom, const Mat& hist2_bottom) {
    // Compute histogram intersection distances for top and bottom halves
    double distance_top = computeChiSquareDistance(hist1_top, hist2_top);
    double distance_bottom = computeChiSquareDistance(hist1_bottom, hist2_bottom);
    
    // Weighted averaging
    double weight_top = 0.5; // Equal weight for top and bottom halves
    double weight_bottom = 0.5;
    return weight_top * distance_top + weight_bottom * distance_bottom;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <target_image_path> <database_dir_path> <N>" << endl;
        return 1;
    }

    // Read target image
    Mat targetImage = imread(argv[1]);
    if (targetImage.empty()) {
        cerr << "Error: Unable to read target image." << endl;
        return 1;
    }

    // Compute RGB histograms for target image
    Mat hist_target_top = computeRGChromaticityHistogram(targetImage(Rect(0, 0, targetImage.cols, targetImage.rows / 2)), 8);
    Mat hist_target_bottom = computeRGChromaticityHistogram(targetImage(Rect(0, targetImage.rows / 2, targetImage.cols, targetImage.rows / 2)), 8);

    // Read database directory
    string databaseDirPath = argv[2];
    int N = atoi(argv[3]);

    // Vector to store distances and file paths
    vector<pair<double, string>> distances;

    // Iterate over images in the database directory
    for (const auto& entry : fs::directory_iterator(databaseDirPath)) {
        // Read image
        Mat image = imread(entry.path().string());
        if (image.empty()) {
            cerr << "Error: Unable to read image " << entry.path().string() << endl;
            continue;
        }

        // Compute RGB histograms for current image
        Mat hist_top = computeRGChromaticityHistogram(image(Rect(0, 0, image.cols, image.rows / 2)), 8);
        Mat hist_bottom = computeRGChromaticityHistogram(image(Rect(0, image.rows / 2, image.cols, image.rows / 2)), 8);

        // Compute multi-histogram distance
        double distance = computeMultiHistogramDistance(hist_target_top, hist_top, hist_target_bottom, hist_bottom);

        // Store distance and file path
        distances.push_back(make_pair(distance, entry.path().string()));
    }

    // Sort images based on distances
    sort(distances.begin(), distances.end());


    // Display the top N images
for (int i = 0; i < min(N, (int)distances.size()); ++i) {
    // Load and display the image
    Mat image = imread(distances[i].second);
    if (!image.empty()) {
        imshow("Image " + to_string(i + 1), image);
        cout << "Distance: " << distances[i].first << ", Image: " << distances[i].second << endl;
    } else {
        cerr << "Error: Unable to read image " << distances[i].second << endl;
    }
}
waitKey(0); // Wait for a key press to exit
destroyAllWindows(); // Close all windows

    return 0;
}
