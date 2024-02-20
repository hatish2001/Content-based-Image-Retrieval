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

// Function to compute color histogram for a given image
Mat computeColorHistogram(const Mat& image, int numBins) {
    Mat hist;

    // Convert image to float
    Mat floatImage;
    image.convertTo(floatImage, CV_32F);

    // Compute color histogram
    vector<Mat> bgr_planes;
    split(floatImage, bgr_planes);
    int histSize[] = { numBins, numBins, numBins };
    float range[] = { 0, 256 };
    const float* ranges[] = { range, range, range };
    int channels[] = { 0, 1, 2 };
    try {
        calcHist(&floatImage, 1, channels, Mat(), hist, 3, histSize, ranges, true, false);
    } catch (const cv::Exception& e) {
        cerr << "Error computing color histogram: " << e.what() << endl;
        return Mat(); // Return empty histogram on error
    }

    // Normalize histogram
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

    return hist;
}

// Function to compute texture histogram for a given image (histogram of gradient orientations and magnitudes)
Mat computeTextureHistogram(const Mat& image, int numBins) {
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Compute gradients
    Mat gx, gy;
    Sobel(gray, gx, CV_32F, 1, 0);
    Sobel(gray, gy, CV_32F, 0, 1);

    // Calculate magnitude and angle
    Mat mag, angle;
    cartToPolar(gx, gy, mag, angle, true);

    // Compute texture histogram (histogram of gradient orientations)
    Mat hist;
    int histSize[] = { numBins };
    float range[] = { 0, 360 };
    const float* ranges[] = { range };
    try {
        calcHist(&angle, 1, 0, Mat(), hist, 1, histSize, ranges, true, false);
    } catch (const cv::Exception& e) {
        cerr << "Error computing texture histogram: " << e.what() << endl;
        return Mat(); // Return empty histogram on error
    }

    // Normalize histogram
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

    return hist;
}

// Function to compute Chi-Square distance between two histograms
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
double computeMultiHistogramDistance(const Mat& hist1_color, const Mat& hist2_color, const Mat& hist1_texture, const Mat& hist2_texture) {
    // Compute Chi-Square distances for color and texture histograms
    double distance_color = computeChiSquareDistance(hist1_color, hist2_color);
    double distance_texture = computeChiSquareDistance(hist1_texture, hist2_texture);
    
    // Equal weighting for color and texture distances
    double weight_color = 0.5;
    double weight_texture = 0.5;
    return weight_color * distance_color + weight_texture * distance_texture;
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

    // Compute color histogram for target image
    Mat hist_target_color = computeColorHistogram(targetImage, 8);

    // Compute texture histogram for target image
    Mat hist_target_texture = computeTextureHistogram(targetImage, 8);

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

        // Compute color histogram for current image
        Mat hist_color = computeColorHistogram(image, 8);

        // Compute texture histogram for current image
        Mat hist_texture = computeTextureHistogram(image, 8);

        // Compute multi-histogram distance
        double distance = computeMultiHistogramDistance(hist_target_color, hist_color, hist_target_texture, hist_texture);

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
