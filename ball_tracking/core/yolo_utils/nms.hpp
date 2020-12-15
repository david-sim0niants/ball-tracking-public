#ifndef NMS_H
#define NMS_H
#include <vector>

struct BBox
{
    float x, y, w, h;
};

typedef std::vector<float> Classes;

void NonMaximumSuppression(std::vector<BBox> &bboxes, std::vector<Classes> &classes_arr, float thresh);

#endif