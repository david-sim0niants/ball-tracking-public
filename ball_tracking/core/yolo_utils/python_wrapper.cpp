#include "nms.hpp"
#include <Python.h>
#include <numpy/arrayobject.h>


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define RAISE_PYERR(err, msg) PyErr_SetString(err, msg); \
        return NULL; \
        
static PyObject* non_maximum_suppression(PyObject* self, PyObject* args)
{
    PyArrayObject *bboxes_np, *classes_arr_np;
    float thresh;
    if (!PyArg_ParseTuple(args, "OOf", &bboxes_np, &classes_arr_np, &thresh))
    {
        return NULL;
    }
    
    float *bboxes_data = reinterpret_cast<float *>(PyArray_DATA(bboxes_np));
    npy_intp *shape = bboxes_np->dimensions;

    BBox *bboxes_obj = new BBox[shape[0]];
    memcpy(bboxes_obj, bboxes_data, shape[0] * sizeof(BBox));
    std::vector<BBox> bboxes(bboxes_obj, bboxes_obj + shape[0]);
    
    float *classes_arr_data = reinterpret_cast<float *>(PyArray_DATA(classes_arr_np));
    shape = classes_arr_np->dimensions;
    
    std::vector<Classes> classes_arr(shape[0]);
    for (size_t i = 0; i < shape[0]; i++)
    {
        float *classes_obj = new float[shape[1]];
        memcpy(classes_obj, classes_arr_data + shape[1] * i, shape[1]);
        classes_arr[i] = Classes(classes_obj, classes_obj + shape[1]);
    }
    NonMaximumSuppression(bboxes, classes_arr, thresh);

    memcpy(bboxes_np->data, bboxes.data(), bboxes.size() * sizeof(BBox));
    PyArray_ENABLEFLAGS(bboxes_np, NPY_ARRAY_OWNDATA);
    
    memcpy(classes_arr_np->data, classes_arr.data(), classes_arr.size() * sizeof(Classes));
    PyArray_ENABLEFLAGS(classes_arr_np, NPY_ARRAY_OWNDATA);

    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"non_maximum_suppression", (PyCFunction)non_maximum_suppression, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef yolo_utils = {
    PyModuleDef_HEAD_INIT,
    "yolo_utils",
    "",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_yolo_utils(void)
{
    import_array();
    return PyModule_Create(&yolo_utils);
}