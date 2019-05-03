#ifndef PLATE_HPP
#define PLATE_HPP

#include <opencv2/core/core.hpp>
#include <string>

namespace platercg
{
namespace plate
{

struct Plate
{
    cv::Mat m_imgPlate;
    cv::Mat m_imgGrayscale;
    cv::Mat m_imgThresh;

    cv::RotatedRect m_locationOfPlateInScene;

    std::string m_strChars;
};

}
}

#endif // PLATE_HPP
