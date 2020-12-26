#ifndef NMS_H
#define NMS_H
#include <vector>

struct BBox
{
    double x, y, w, h;
};

typedef std::vector<double> Classes;

void NonMaximumSuppression(std::vector<BBox> &bboxes, std::vector<Classes> &classes_arr, double thresh);

#endif