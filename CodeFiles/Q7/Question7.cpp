/*

Authored by: Aadhi Aadhavan Balasubramanian, Harishraj Udaya Bhaskar
Date: 01/26/2024

*/

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

// Function to compute cosine distance between two feature vectors
double computeCosineDistance(const Mat& vec1, const Mat& vec2) {
    double dotProduct = vec1.dot(vec2);
    double normVec1 = norm(vec1);
    double normVec2 = norm(vec2);
    return 1.0 - dotProduct / (normVec1 * normVec2);
}

// Function to compute histogram intersection distance between two histograms
double computeChiSquaredDistance(const Mat& hist1, const Mat& hist2) {
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

    Mat floatImage;
    image.convertTo(floatImage, CV_32F);

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
        return Mat();
    }

    normalize(rgHist, hist, 0, 1, NORM_MINMAX, -1, Mat());

    return hist;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " <feature_vectors_csv_path> <target_image_path> <database_dir> <N>" << endl;
        return 1;
    }

    string csvFilePath = argv[1];
    string targetImagePath = argv[2];
    string databaseDir = argv[3];
    int N = atoi(argv[4]);

    // Read target image's feature vector from the CSV file
    Mat targetFeatureVector;
    ifstream csvFile(csvFilePath);
    if (!csvFile.is_open()) {
        cerr << "Error: Unable to open CSV file." << endl;
        return 1;
    }

    string line;
    while (getline(csvFile, line)) {
        istringstream iss(line);
        vector<string> tokens;
        string token;
        while (getline(iss, token, ',')) {
            tokens.push_back(token);
        }
        if (tokens.size() < 2) {
            cerr << "Error: Invalid CSV format." << endl;
            return 1;
        }
        if (tokens[0] == fs::path(targetImagePath).filename()) {
            tokens.erase(tokens.begin());
            vector<float> values;
            for (const string& val : tokens) {
                values.push_back(stof(val));
            }
            targetFeatureVector = Mat(values, true).reshape(1, 1);
            break;
        }
    }
    csvFile.close();
    if (targetFeatureVector.empty()) {
        cerr << "Error: Feature vector not found for target image." << endl;
        return 1;
    }

    // Read target image
    Mat targetImage = imread(targetImagePath);
    if (targetImage.empty()) {
        cerr << "Error: Unable to read target image." << endl;
        return 1;
    }

    // Compute RG chromaticity histogram for the target image
    Mat targetHist = computeRGChromaticityHistogram(targetImage, 16);
    if (targetHist.empty()) {
        cerr << "Error: Unable to compute histogram for the target image." << endl;
        return 1;
    }

    // Read feature vectors for database images from the CSV file and compute distances
    vector<pair<double, string>> distances;
    csvFile.open(csvFilePath);
    if (!csvFile.is_open()) {
        cerr << "Error: Unable to open CSV file." << endl;
        return 1;
    }
    while (getline(csvFile, line)) {
        istringstream iss(line);
        vector<string> tokens;
        string token;
        while (getline(iss, token, ',')) {
            tokens.push_back(token);
        }
        if (tokens.size() < 513) {
            cerr << "Error: Invalid CSV format." << endl;
            return 1;
        }
        string filename = tokens[0];
        tokens.erase(tokens.begin());
        vector<float> values;
        for (const string& val : tokens) {
            values.push_back(stof(val));
        }
        Mat featureVector = Mat(values, true).reshape(1, 1);

        // Compute cosine distance between feature vectors
        double featureDistance = computeCosineDistance(targetFeatureVector, featureVector);

        // Read and compute histogram for the current image
        Mat image = imread(databaseDir + "/" + filename);
        if (image.empty()) {
            cerr << "Error: Unable to read image " << filename << endl;
            continue;
        }
        Mat imageHist = computeRGChromaticityHistogram(image, 16);
        if (imageHist.empty()) {
            cerr << "Error: Unable to compute histogram for image " << filename << endl;
            continue;
        }

        // Compute histogram intersection distance between target and current image
        double histDistance = computeChiSquaredDistance(targetHist, imageHist);

        // Combine distances using a weighted average or other strategies as needed
        double combinedDistance = (featureDistance + histDistance) / 2.0;

        distances.push_back(make_pair(combinedDistance, filename));
    }
    csvFile.close();

    // Sort images based on combined distances
    sort(distances.begin(), distances.end());

    // Display the top N images
    for (int i = 0; i < min(N, static_cast<int>(distances.size())); ++i) {
        cout << "Distance: " << distances[i].first << ", Image: " << distances[i].second << endl;
    }

    return 0;
}
