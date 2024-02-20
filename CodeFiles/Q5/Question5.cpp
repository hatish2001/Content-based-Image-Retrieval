/*

Authored by: Aadhi Aadhavan Balasubramanian, Harishraj Udaya Bhaskar
Date: 01/26/2024

*/

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

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

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <feature_vectors_csv_path> <target_image_filename> <N>" << endl;
        return 1;
    }

    // Read target image's feature vector from the CSV file
    string csvFilePath = argv[1];
    string targetImageFilename = argv[2];
    int N = atoi(argv[3]);

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
        if (tokens[0] == targetImageFilename) {
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
        double distance = computeCosineDistance(targetFeatureVector, featureVector);
        distances.push_back(make_pair(distance, filename));
    }
    csvFile.close();

    // Sort images based on distances
    sort(distances.begin(), distances.end());

    // Display the top N images
    for (int i = 0; i < min(N, static_cast<int>(distances.size())); ++i) {
        cout << "Distance: " << distances[i].first << ", Image: " << distances[i].second << endl;
    }

    return 0;
}
