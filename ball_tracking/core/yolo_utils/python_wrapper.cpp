#include "nms.hpp"
#include <Python.h>
#include <numpy/arrayobject.h>


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define RAISE_PYERR(err, msg) PyErr_SetString(err, msg); \
        return NULL; \
        
static PyObject* non_maximum_suppression(PyObject* self, PyObject* args)
{
    PyArrayObject *bboxes_np, *classes_arr_np;
    double thresh;
    if (!PyArg_ParseTuple(args, "OOd", &bboxes_np, &classes_arr_np, &thresh))
    {
        return NULL;
    }
    
    double *bboxes_data = reinterpret_cast<double *>(PyArray_DATA(bboxes_np));
    npy_intp *shape = bboxes_np->dimensions;

    BBox *bboxes_obj = new BBox[shape[0]];
    memcpy(bboxes_obj, bboxes_data, shape[0] * sizeof(BBox));
    std::vector<BBox> bboxes(bboxes_obj, bboxes_obj + shape[0]);
    
    double *classes_arr_data = reinterpret_cast<double *>(PyArray_DATA(classes_arr_np));
    shape = classes_arr_np->dimensions;
    std::vector<Classes> classes_arr(shape[0]);
    for (size_t i = 0; i < shape[0]; i++)
    {
        double *classes_obj = new double[shape[1]];
        memcpy(classes_obj, classes_arr_data + shape[1] * i, shape[1] * sizeof(double));
        classes_arr[i] = Classes(classes_obj, classes_obj + shape[1]);
    }
    
    NonMaximumSuppression(bboxes, classes_arr, thresh);
    
    for (size_t i = 0; i < classes_arr.size(); i++)
    {
        memcpy(classes_arr_np->data + i * classes_arr[i].size() * sizeof(double), classes_arr[i].data(), classes_arr[i].size() * sizeof(double));
    }
    
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