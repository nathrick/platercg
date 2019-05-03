#include "../include/platercg/character.hpp"
#include "../include/platercg/KnnStrategy.hpp"
#include "../include/platercg/plate.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>

namespace platercg
{
constexpr auto PLATE_WIDTH_PADDING_FACTOR = 1.3;
constexpr auto PLATE_HEIGHT_PADDING_FACTOR = 1.5;

const cv::Size GAUSSIAN_SMOOTH_FILTER_SIZE = cv::Size(5, 5);
constexpr auto ADAPTIVE_THRESH_BLOCK_SIZE = 19;
constexpr auto ADAPTIVE_THRESH_WEIGHT = 9;

constexpr auto MIN_NUMBER_OF_MATCHING_CHARS = 5;

constexpr auto RESIZED_CHAR_IMAGE_WIDTH = 20;
constexpr auto RESIZED_CHAR_IMAGE_HEIGHT = 30;

constexpr auto MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3;

const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 255.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

using Char = character::Character;
using CharVect = std::vector<Char>;

class KnnStrategy::Impl
{
public:
    Impl();
    bool trainKNN();
    bool loadKNNData(const std::string &lblFile, const std::string &datFile);

    void preprocess(cv::Mat &imgOriginal, cv::Mat &imgGrayscale, cv::Mat &imgThresh);
    cv::Mat extractValue(cv::Mat &imgOriginal);
    cv::Mat maximizeContrast(cv::Mat &imgGrayscale);

    CharVect findChars(cv::Mat &imgTresh);
    CharVect findMatchingChars(const character::Character& inputChar, CharVect &inputChars);
    std::vector<CharVect> findGroupsOfMatchingChars(CharVect& inputChars);

    plate::Plate extractPlate(cv::Mat &imgOriginal, CharVect &chars);
    void removeOverlappingChars(CharVect& charGroup);
    std::string recognizeCharsInPlate(cv::Mat &imgThresh, CharVect &charGroup);

private:
    cv::Mat m_trainData;
    cv::Mat m_trainLabels;

    cv::Ptr<cv::ml::KNearest> m_kNearest;
};

KnnStrategy::Impl::Impl()
{
    m_kNearest = cv::ml::KNearest::create();

    std::cout << "knearest created..." << std::endl;
}

bool KnnStrategy::Impl::trainKNN()
{
    m_kNearest->setDefaultK(1);
    return m_kNearest->train(m_trainData, cv::ml::ROW_SAMPLE, m_trainLabels);
}

bool KnnStrategy::Impl::loadKNNData(const std::string &lblFile, const std::string &datFile)
{
    cv::FileStorage labelsFile(lblFile.c_str(), cv::FileStorage::READ);

    if(!labelsFile.isOpened())
    {
        std::cout << "labels file cannot be opened !!!" << std::endl;
        return false;
    }

    labelsFile["labels"] >> m_trainLabels;
    labelsFile.release();

    cv::FileStorage dataFile(datFile.c_str(), cv::FileStorage::READ);

    if(!dataFile.isOpened())
    {
        std::cout << "data file cannot be opened !!!" << std::endl;
        return false;
    }

    dataFile["data"] >> m_trainData;
    dataFile.release();

    return true;
}

void KnnStrategy::Impl::preprocess(cv::Mat &imgOriginal, cv::Mat &imgGrayscale, cv::Mat &imgThresh)
{
    imgGrayscale = extractValue(imgOriginal);
    cv::Mat imgMaxContrastGrayscale = maximizeContrast(imgGrayscale);

    cv::Mat imgBlurred;
    cv::GaussianBlur(imgMaxContrastGrayscale,
                     imgBlurred,
                     GAUSSIAN_SMOOTH_FILTER_SIZE,
                     0);

    cv::adaptiveThreshold(imgBlurred,
                          imgThresh,
                          255.0,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY_INV,
                          ADAPTIVE_THRESH_BLOCK_SIZE,
                          ADAPTIVE_THRESH_WEIGHT);
}

cv::Mat KnnStrategy::Impl::extractValue(cv::Mat &imgOriginal)
{
    cv::Mat imgHSV;
    std::vector<cv::Mat> vectorOfHSVImages;

    cv::cvtColor(imgOriginal, imgHSV, cv::COLOR_BGR2HSV);
    cv::split(imgHSV, vectorOfHSVImages);

    return vectorOfHSVImages[2];
}

cv::Mat KnnStrategy::Impl::maximizeContrast(cv::Mat &imgGrayscale)
{
    cv::Mat imgTopHat;
    cv::Mat imgBlackHat;
    cv::Mat imgGrayscalePlusTopHat;
    cv::Mat imgGrayscalePlusTopHatMinusBlackHat;

    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT,
                                                           cv::Size(3, 3));

    cv::morphologyEx(imgGrayscale, imgTopHat, cv::MORPH_TOPHAT, structuringElement);
    cv::morphologyEx(imgGrayscale, imgBlackHat, cv::MORPH_BLACKHAT, structuringElement);

    imgGrayscalePlusTopHat = imgGrayscale + imgTopHat;
    imgGrayscalePlusTopHatMinusBlackHat = imgGrayscalePlusTopHat - imgBlackHat;

    return imgGrayscalePlusTopHatMinusBlackHat;
}

CharVect KnnStrategy::Impl::findChars(cv::Mat &imgTresh)
{
    CharVect result;

    cv::Mat imgContours(imgTresh.size(), CV_8UC3, SCALAR_BLACK);
    cv::Mat imgThreshCopy = imgTresh.clone();

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(imgThreshCopy, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    for(auto &c : contours)
    {
        character::Character possibleChar(c);

        if(possibleChar.firstPassCheck())
        {
            result.push_back(possibleChar);
        }
    }

    return result;
}

CharVect KnnStrategy::Impl::findMatchingChars(const character::Character &inputChar, CharVect &inputChars)
{
    CharVect result;

    if(!inputChar.alreadyMatch())
    {
        for(auto &matchingCandidate : inputChars)
        {
            if((matchingCandidate != inputChar) && (!matchingCandidate.alreadyMatch()))
            {
                if(inputChar.isMatchingChar(matchingCandidate))
                {
                    matchingCandidate.setAlreadyMatch(true);
                    result.push_back(matchingCandidate);
                }
            }
        }
    }

    return result;
}

std::vector<CharVect> KnnStrategy::Impl::findGroupsOfMatchingChars(CharVect &inputChars)
{
    std::vector<CharVect> result;

    for(auto &inputChar : inputChars)
    {
        if(!inputChar.alreadyMatch())
        {
            CharVect matchingChars = findMatchingChars(inputChar, inputChars);

            /* Set input char as a match and add to a group */
            inputChar.setAlreadyMatch(true);
            matchingChars.push_back(inputChar);

            if(matchingChars.size() < MIN_NUMBER_OF_MATCHING_CHARS)
            {
                continue;
            }

            result.push_back(matchingChars);
        }
    }

    return result;
}

plate::Plate KnnStrategy::Impl::extractPlate(cv::Mat &imgOriginal, CharVect &chars)
{
    plate::Plate resultPlate;

    std::sort(chars.begin(), chars.end(),[](const character::Character& first, const character::Character& second)
    {
        return first.centerX() < second.centerX();
    });

    double plateCenterX = static_cast<double>(chars[0].centerX() + chars[chars.size() - 1].centerX()) / 2.0;
    double plateCenterY = static_cast<double>(chars[0].centerY() + chars[chars.size() - 1].centerY()) / 2.0;
    cv::Point2d plateCenter(plateCenterX, plateCenterY);

    int plateWidth = static_cast<int>((chars[chars.size() - 1].boundingRect().x +
                                       chars[chars.size() - 1].boundingRect().width -
                                       chars[0].boundingRect().x) * PLATE_WIDTH_PADDING_FACTOR);

    double charHeightSum = 0;
    for(auto &matchingChar : chars)
    {
        charHeightSum += matchingChar.boundingRect().height;
    }

    double averageCharHeight = charHeightSum / static_cast<double>(chars.size());
    double plateHeight = averageCharHeight * PLATE_HEIGHT_PADDING_FACTOR;

    double opposite = chars[chars.size() - 1].centerY() - chars[0].centerY();
    double hypotenuse = chars[0].distance(chars[chars.size() - 1]);
    double correctionAngleInRad = asin(opposite / hypotenuse);
    double correctionAngleInDeg = correctionAngleInRad * (180.0 / CV_PI);

    resultPlate.m_locationOfPlateInScene =
        cv::RotatedRect(plateCenter,
                        cv::Size2f(static_cast<float>(plateWidth), static_cast<float>(plateHeight)),
                        static_cast<float>(correctionAngleInDeg));

    cv::Mat rotationMatrix = cv::getRotationMatrix2D(plateCenter, correctionAngleInDeg, 1.0);
    cv::Mat imgRotated;
    cv::Mat imgCropped;

    cv::warpAffine(imgOriginal, imgRotated, rotationMatrix, imgOriginal.size());

    cv::getRectSubPix(imgRotated,
                      resultPlate.m_locationOfPlateInScene.size,
                      resultPlate.m_locationOfPlateInScene.center,
                      imgCropped);

    resultPlate.m_imgPlate = imgCropped;

    return resultPlate;
}

void KnnStrategy::Impl::removeOverlappingChars(CharVect &charGroup)
{
    for(auto& ch : charGroup)
    {
        for(auto& k : charGroup)
        {
            if(ch != k)
            {
                if(ch.distance(k) < (ch.diagonalSize() * MIN_DIAG_SIZE_MULTIPLE_AWAY))
                {
                    if(ch.boundingRect().area() < k.boundingRect().area())
                    {
                        ch.setOverlappingChar(true);
                    }
                    else
                    {
                        k.setOverlappingChar(true);
                    }
                }
            }
        }
    }

    charGroup.erase(std::remove_if(charGroup.begin(), charGroup.end(), [](const character::Character& ch){
        return ch.overlappingChar();
    }), charGroup.end());
}

std::string KnnStrategy::Impl::recognizeCharsInPlate(cv::Mat &imgThresh, CharVect &charGroup)
{
    std::string strChars;

    for (auto &currentChar : charGroup)
    {
        cv::Mat imgROItoBeCloned = imgThresh(currentChar.boundingRect()); // get ROI image of bounding rect
        cv::Mat imgROI = imgROItoBeCloned.clone();          // clone ROI image so we don't change original when we resize

        cv::Mat imgROIResized;
        // resize image, this is necessary for char recognition
        cv::resize(imgROI, imgROIResized, cv::Size(RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT));

        cv::Mat matROIFloat;
        imgROIResized.convertTo(matROIFloat, CV_32FC1); // convert Mat to float, necessary for call to findNearest
        cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1); // flatten Matrix into one row

        cv::Mat matCurrentChar(0, 0, CV_32F);       // declare Mat to read current char into, this is necessary
                                                    // b/c findNearest requires a Mat

        m_kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar); // finally we can call find_nearest !!!
        float fltCurrentChar = matCurrentChar.at<float>(0, 0); // convert current char from Mat to float

        strChars += char(int(fltCurrentChar)); // append current char to full string
    }

    return strChars;
}


/* KNN_Rcg */

KnnStrategy::KnnStrategy(const std::string &lblFile, const std::string &datFile) :
    m_labelsFile(lblFile),
    m_dataFile(datFile),
    m_pImpl(std::make_unique<Impl>())
{
    if (m_pImpl.get()->loadKNNData(lblFile, datFile))
    {
        if(!(m_pImpl.get()->trainKNN()))
        {
            std::cout << "KNN training failed..." << std::endl;
        }
    }
    else
    {
        std::cout << "Loading KNN data failed..." << std::endl;
    }

}

KnnStrategy::~KnnStrategy() = default;

std::vector<plate::Plate> KnnStrategy::licensePlates(cv::Mat frame) const
{
    auto plates = possiblePlates(frame);

    for(auto& plate : plates)
    {
        m_pImpl.get()->preprocess(plate.m_imgPlate, plate.m_imgGrayscale, plate.m_imgThresh);

        cv::resize(plate.m_imgThresh, plate.m_imgThresh, cv::Size(), 1.6, 1.6);
        cv::threshold(plate.m_imgThresh,
                      plate.m_imgThresh,
                      0.0,
                      255.0,
                      cv::THRESH_BINARY | cv::THRESH_OTSU);

        CharVect possibleChars = m_pImpl.get()->findChars(plate.m_imgThresh);

        std::vector<CharVect> groupsOfMatchingChar = m_pImpl.get()->findGroupsOfMatchingChars(possibleChars);
        if(groupsOfMatchingChar.empty())
        {
            plate.m_strChars = "";
            continue;
        }

        for(auto& charGroup : groupsOfMatchingChar)
        {
            std::sort(charGroup.begin(), charGroup.end(),
                      [](const character::Character& leftChar, const character::Character& rightChar){
                return leftChar.centerX() < rightChar.centerX();
            });

            m_pImpl.get()->removeOverlappingChars(charGroup);
        }

        auto longestGroup = *(std::max_element(groupsOfMatchingChar.begin(),
                                               groupsOfMatchingChar.end(),
                                               [](const auto& g1, const auto& g2)
        {
            return g1.size() < g2.size();
        }));

        plate.m_strChars = m_pImpl.get()->recognizeCharsInPlate(plate.m_imgThresh, longestGroup);
    }

    return plates;
}

std::vector<plate::Plate> KnnStrategy::possiblePlates(cv::Mat frame) const
{
    std::vector<plate::Plate> plates;

    cv::Mat imgGrayscale;
    cv::Mat imgThresh;

    m_pImpl.get()->preprocess(frame, imgGrayscale, imgThresh);

    CharVect possibleCharacters = m_pImpl.get()->findChars(imgThresh);

    std::vector<CharVect> groupsOfMatchingChar = m_pImpl.get()->findGroupsOfMatchingChars(possibleCharacters);

    for(auto &group : groupsOfMatchingChar)
    {
        plate::Plate possiblePlate = m_pImpl.get()->extractPlate(frame, group);

        if(!possiblePlate.m_imgPlate.empty())
        {
            plates.push_back(possiblePlate);
        }
    }

    return plates;
}
} // namespace platercg
