#include "../include/platercg/character.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace platercg
{
namespace character
{
constexpr auto MIN_PIXEL_WIDTH = 2;
constexpr auto MIN_PIXEL_HEIGHT = 8;
constexpr auto MIN_ASPECT_RATIO = 0.25;
constexpr auto MAX_ASPECT_RATIO = 1.0;
constexpr auto MIN_PIXEL_AREA = 80;

constexpr auto MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3;
constexpr auto MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0;
constexpr auto MAX_CHANGE_IN_AREA = 0.5;
constexpr auto MAX_CHANGE_IN_WIDTH = 0.8;
constexpr auto MAX_CHANGE_IN_HEIGHT = 0.2;
constexpr auto MAX_ANGLE_BETWEEN_CHARS = 12.0;

platercg::character::Character::Character(std::vector<cv::Point> & contour) :
    m_contour(contour),
    m_alreadyMatch(false)
{
    m_boundingRect = cv::boundingRect(m_contour);

    m_centerX = (m_boundingRect.x + m_boundingRect.x + m_boundingRect.width) / 2;
    m_centerY = (m_boundingRect.y + m_boundingRect.y + m_boundingRect.height) / 2;

    m_diagonalSize = sqrt(pow(m_boundingRect.width, 2) + pow(m_boundingRect.height, 2));
    m_aspectRatio = static_cast<double>(m_boundingRect.width) /
                    static_cast<double>(m_boundingRect.height);
}

bool platercg::character::Character::operator==(const Character &otherChar) const
{
    return m_contour == otherChar.m_contour;
}

bool platercg::character::Character::operator!=(const Character &otherChar) const
{
    return m_contour != otherChar.m_contour;
}

bool platercg::character::Character::firstPassCheck() const
{
    return m_boundingRect.area() > MIN_PIXEL_AREA &&
           m_boundingRect.width > MIN_PIXEL_WIDTH &&
           m_boundingRect.height > MIN_PIXEL_HEIGHT &&
           m_aspectRatio > MIN_ASPECT_RATIO &&
           m_aspectRatio < MAX_ASPECT_RATIO;
}

bool platercg::character::Character::isMatchingChar(const Character &otherChar) const
{
    double changeInArea =
        static_cast<double>(abs(m_boundingRect.area()) - otherChar.m_boundingRect.area()) /
        static_cast<double>(otherChar.m_boundingRect.area());
    double changeInWidth =
        static_cast<double>(abs(m_boundingRect.width - otherChar.m_boundingRect.width)) /
        static_cast<double>(otherChar.m_boundingRect.width);
    double changeInHeight =
        static_cast<double>(abs(m_boundingRect.height - otherChar.m_boundingRect.height)) /
        static_cast<double>(otherChar.m_boundingRect.height);

    return (distance(otherChar) <
            (otherChar.m_diagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY)) &&
           (angle(otherChar) < MAX_ANGLE_BETWEEN_CHARS) &&
           (changeInArea < MAX_CHANGE_IN_AREA) &&
           (changeInWidth < MAX_CHANGE_IN_WIDTH) &&
           (changeInHeight < MAX_CHANGE_IN_HEIGHT);
}

double platercg::character::Character::distance(const Character &otherChar) const
{
    double absX = abs(m_centerX - otherChar.m_centerX);
    double absY = abs(m_centerY - otherChar.m_centerY);

    return sqrt(pow(absX, 2) + pow(absY, 2));
}

double platercg::character::Character::angle(const Character &otherChar) const
{
    double absX = abs(m_centerX - otherChar.m_centerX);
    double absY = abs(m_centerY - otherChar.m_centerY);

    double angleRad = atan(absY / absX);
    double angleDeg = angleRad * (180.0 / CV_PI);

    return angleDeg;
}

std::vector<cv::Point> platercg::character::Character::contour() const
{
    return m_contour;
}

cv::Rect platercg::character::Character::boundingRect() const
{
    return m_boundingRect;
}

int platercg::character::Character::centerX() const
{
    return m_centerX;
}

int platercg::character::Character::centerY() const
{
    return m_centerY;
}

double platercg::character::Character::diagonalSize() const
{
    return m_diagonalSize;
}

double platercg::character::Character::aspectRatio() const
{
    return m_aspectRatio;
}

bool platercg::character::Character::alreadyMatch() const
{
    return m_alreadyMatch;
}

void platercg::character::Character::setAlreadyMatch(bool alreadyMatch)
{
    m_alreadyMatch = alreadyMatch;
}

bool platercg::character::Character::overlappingChar() const
{
    return m_overlappingChar;
}

void platercg::character::Character::setOverlappingChar(bool overlappingChar)
{
    m_overlappingChar = overlappingChar;
}
}
}
