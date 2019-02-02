#ifndef KNN_RCG_HPP
#define KNN_RCG_HPP

#include "character.hpp"
#include "plate.hpp"

#include <memory>
#include <opencv2/core/core.hpp>
#include <string>

namespace platercg
{
class KNN_Rcg
{
public:
    KNN_Rcg(const std::string& lblFile, const std::string &datFile);
    ~KNN_Rcg();

    std::vector<plate::Plate> licensePlates(cv::Mat frame) const;

private:
    std::string m_labelsFile;
    std::string m_dataFile;

    class Impl;
    std::unique_ptr<Impl> m_pImpl;

    std::vector<plate::Plate> possiblePlates(cv::Mat frame) const;
};
}

#endif // KNN_RCG_HPP
