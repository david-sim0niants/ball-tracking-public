#include "nms.hpp"
#include <vector>
#include <algorithm>
#include <numeric>


template <typename T>
std::vector<size_t> argsort(const std::vector<T> &v) 
{
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    return idx;
}

float interval_overlap(float a0, float b0, float a1, float b1)
{
    if (a1 < a0)
    {
        if (b1 < a0)
            return 0;
        else return std::min(b0, b1) - a0;
    }
    else
    {
        if (b0 < a1)
            return 0;
        else return std::min(b0, b1) - b1;
    }   
}

float bbox_iou(BBox bbox_a, BBox bbox_b)
{
    float intersect_w = interval_overlap(bbox_a.x, bbox_a.x + bbox_a.h, bbox_b.x, bbox_b.x + bbox_b.h);
    float intersect_h = interval_overlap(bbox_a.y, bbox_a.y + bbox_a.h, bbox_b.y, bbox_b.y + bbox_b.h);
    
    float intersect = intersect_w * intersect_h;
    float _union = bbox_a.w * bbox_a.h + bbox_b.w * bbox_b.h - intersect;
    
    return intersect / _union;
}

void NonMaximumSuppression(std::vector<BBox> &bboxes, std::vector<Classes> &classes_arr, float thresh)
{
    if (bboxes.size() != classes_arr.size() || classes_arr.size() == 0)
        return;
    int num_classes = classes_arr[0].size();
    
    for (int c = 0; c < num_classes; c++)
    {
        std::vector<float> classes_along_bboxes(bboxes.size());
        for (size_t b = 0; b < bboxes.size(); b++)
        {
            classes_along_bboxes[b] = classes_arr[b][c];
        }
        
        std::vector<size_t> sorted_indices = argsort(classes_along_bboxes);
        for (size_t i = 0; i < bboxes.size(); i++)
        {
            size_t index_i = sorted_indices[i];
            if (classes_arr[index_i][c] == 0) continue;
            for (size_t j = i + 1; j < bboxes.size(); j++)
            {
                size_t index_j = sorted_indices[j];
                if (bbox_iou(bboxes[index_i], bboxes[index_j]) >= thresh)
                    classes_arr[index_j][c] = 0;
            }
        }
    }
}
