#ifndef ISTRATEGY_HPP
#define ISTRATEGY_HPP

#include "Plate.hpp"

namespace platercg
{

using PlateVec = std::vector<plate::Plate>;

class IStrategy
{
public:
    IStrategy() = default;
    virtual ~IStrategy() = default;

    virtual PlateVec licensePlates(cv::Mat frame) const = 0;


};

}
#endif
