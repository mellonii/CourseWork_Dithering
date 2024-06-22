#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <ctime>

cv::Mat Bayer_dithering(cv::Mat image) {

    cv::Mat bw_image;
    cv::cvtColor(image, bw_image, cv::COLOR_BGR2GRAY);

    int bayerMatrix[4][4] = {
        {  0,  8,  2, 10 },
        { 12,  4, 14,  6 },
        {  3, 11,  1,  9 },
        { 15,  7, 13,  5 }
    };

    cv::Mat ditherMatrix(4, 4, CV_8UC1);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            ditherMatrix.at<uchar>(i, j) = cv::saturate_cast<uchar>((bayerMatrix[i][j] / 16.0) * 255);
        }
    }

    cv::Mat result = bw_image.clone();
    for (int y = 0; y < bw_image.rows; ++y) {
        for (int x = 0; x < bw_image.cols; ++x) {
            uchar pixelValue = bw_image.at<uchar>(y, x);
            uchar ditherValue = ditherMatrix.at<uchar>(y % 4, x % 4);
            result.at<uchar>(y, x) = (pixelValue > ditherValue) ? 255 : 0;
        }
    }
    return result;
}

cv::Mat Random_dithering(cv::Mat image) {

    cv::Mat bw_image;
    cv::cvtColor(image, bw_image, cv::COLOR_BGR2GRAY);

    cv::Mat result;
    bw_image.copyTo(result);

    std::srand(std::time(0));

    for (int y = 0; y < result.rows; ++y) {
        for (int x = 0; x < result.cols; ++x) {

            float noise = static_cast<float>(rand()) / RAND_MAX - 0.5;
            uchar newPixel = (result.at<uchar>(y, x) / 255.0 + noise) > 0.5 ? 255 : 0;
            result.at<uchar>(y, x) = newPixel;
        }
    }

    return result;
}

cv::Mat FloydSteinberg_method(cv::Mat image) {

    cv::Mat bw_image;
    cv::cvtColor(image, bw_image, cv::COLOR_BGR2GRAY);

    cv::Mat result;
    bw_image.copyTo(result);

    for (int y = 0; y < image.rows - 1; ++y) {
        for (int x = 1; x < image.cols - 1; ++x) {
            uchar oldPixel = result.at<uchar>(y, x);
            uchar newPixel = oldPixel > 128 ? 255 : 0;
            result.at<uchar>(y, x) = newPixel;

            int error = oldPixel - newPixel;

            result.at<uchar>(y, x + 1) += error * 7 / 16.0;
            result.at<uchar>(y + 1, x - 1) += error * 3 / 16.0;
            result.at<uchar>(y + 1, x) += error * 5 / 16.0;
            result.at<uchar>(y + 1, x + 1) += error * 1 / 16.0;
        }
    }

    return result;
}

cv::Mat generateBlueNoise(int width, int height) {

    cv::Mat blueNoise = cv::Mat::zeros(height, width, CV_32F);
    std::srand(std::time(0));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            blueNoise.at<float>(y, x) = static_cast<float>(std::rand()) / RAND_MAX;
        }
    }

    cv::GaussianBlur(blueNoise, blueNoise, cv::Size(7, 7), 1.5);
    cv::normalize(blueNoise, blueNoise, 0.0f, 1.0f, cv::NORM_MINMAX);

    return blueNoise;
}

cv::Mat BlueNoise_dithering(cv::Mat image) {

    cv::Mat blueNoise = generateBlueNoise(64, 64);

    cv::Mat bw_image;
    cv::cvtColor(image, bw_image, cv::COLOR_BGR2GRAY);

    cv::Mat result;
    bw_image.copyTo(result);

    for (int y = 0; y < result.rows; ++y) {
        for (int x = 0; x < result.cols; ++x) {
            float brightness = result.at<uchar>(y, x) / 255.0f;
            float noise = blueNoise.at<float>(y % blueNoise.rows, x % blueNoise.cols);

            uchar newPixel = (brightness + noise - 0.5f) > 0.5f ? 255 : 0;
            result.at<uchar>(y, x) = newPixel;
        }
    }

    return result;
}

void Dithering(const std::string imagePath, const int method) {

    std::cout << "Image path: " << imagePath << std::endl;
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Test Image Error" << std::endl;
        return;
    }

    cv::Mat result;
    std::string method_name;

    switch (method) {
    case 1: result = Bayer_dithering(image); method_name = "Bayer Dithering";  break;
    case 2: result = Random_dithering(image); method_name = "Random Dithering";   break;
    case 3: result = FloydSteinberg_method(image); method_name = "Floyd Steinberg Dithering";   break;
    case 4: result = BlueNoise_dithering(image); method_name = "Blue Noise Dithering";   break;
    case 5: {
        cv::imshow("Image", image);
        result = Bayer_dithering(image); method_name = "Bayer Dithering";
        cv::imshow("Result of " + method_name, result);
        result = Random_dithering(image); method_name = "Random Dithering";
        cv::imshow("Result of " + method_name, result);
        result = FloydSteinberg_method(image); method_name = "Floyd Steinberg Dithering";
        cv::imshow("Result of " + method_name, result);
        result = BlueNoise_dithering(image); method_name = "Blue Noise Dithering";
        cv::imshow("Result of " + method_name, result);
        cv::waitKey(0);
        return;
    }
    }
    cv::imshow("Image", image);
    cv::imshow("Result of " + method_name, result);
    cv::waitKey(0);
}

int main() {
    std::string imagePath;
    int method = 5;

    //imagePath = "main_test.jpg";
    //imagePath = "test.jpg";
    //imagePath = "test2.jpg";
    imagePath = "test3.jpg";
    Dithering(imagePath, method);
    return 0;
}
