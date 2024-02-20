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

// Function to compute features for a given image
Mat computeFeatures(const Mat& image) {
    // Extract the 7x7 square in the middle of the image as the feature vector
    Rect roi((image.cols - 7) / 2, (image.rows - 7) / 2, 7, 7);
    Mat featureVector = image(roi).clone().reshape(1, 1);

    return featureVector;
}

// Function to compute sum-of-squared-difference distance between two feature vectors
double computeDistance(const Mat& ft, const Mat& fi) {
    Mat diff;
    absdiff(ft, fi, diff);
    diff.convertTo(diff, CV_64F);
    return sum(diff.mul(diff))[0];
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

    // Compute features for the target image
    Mat ft = computeFeatures(targetImage);

    // Loop over the directory of images
    vector<pair<double, string>> matches; // (distance, image_path) pairs
    for (const auto& entry : fs::directory_iterator(databaseDir)) {
        string imagePath = entry.path().string();
        Mat image = imread(imagePath);
        if (image.empty()) {
            cerr << "Error: Unable to read image " << imagePath << endl;
            continue;
        }

        // Compute features for the current image
        Mat fi = computeFeatures(image);

        // Compute distance between target image and current image
        double distance = computeDistance(ft, fi);

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
