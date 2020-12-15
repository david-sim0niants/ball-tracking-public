from distutils.core import setup, Extension
from numpy import get_include
import os
import shutil


if __name__ == '__main__':
    path_to_sources = os.path.abspath("ball_tracking/core/yolo_utils")
    sources = ['python_wrapper.cpp', 'nms.cpp']
    for i in range(len(sources)):
        sources[i] = os.path.join(path_to_sources, sources[i])
    
    module = Extension("yolo_utils", sources=sources, include_dirs=[get_include()])

    setup(name='yolo_utils', version='1.0', 
    description='', ext_modules=[module])


    file_found = False
    if os.path.exists('build'):
        for directory in os.listdir('build'):
            directory = os.path.join('build', directory)
            if os.path.isdir(directory):
                for file in os.listdir(directory):
                    fp = os.path.join(directory, file)
                    if os.path.isfile(fp):
                        if file.endswith('.pyd') or file.endswith('.so') or file.endswith('.dll'):
                            os.replace(fp, os.path.join('ball_tracking/core', file))
                            file_found = True
                            break
            if file_found:
                break
        shutil.rmtree('build')