#ifndef CHARACTER_HPP
#define CHARACTER_HPP

#include <opencv2/core/core.hpp>

namespace platercg
{
namespace character
{
    class Character
    {
    public:
        Character(std::vector<cv::Point> &contour);
        bool operator==(const Character& otherChar) const;
        bool operator!=(const Character& otherChar) const;

        bool firstPassCheck() const;
        bool isMatchingChar(const Character& otherChar) const;

        double distance(const Character &otherChar) const;
        double angle(const Character &otherChar) const;
        std::vector<cv::Point> contour() const;
        cv::Rect boundingRect() const;
        int centerX() const;
        int centerY() const;
        double diagonalSize() const;
        double aspectRatio() const;

        bool alreadyMatch() const;
        void setAlreadyMatch(bool alreadyMatch);
        bool overlappingChar() const;
        void setOverlappingChar(bool overlappingChar);

    private:
        std::vector<cv::Point> m_contour;
        cv::Rect m_boundingRect;
        int m_centerX;
        int m_centerY;
        double m_diagonalSize;
        double m_aspectRatio;

        bool m_alreadyMatch;
        bool m_overlappingChar;
    };
}
}

#endif // CHARACTER_HPP
