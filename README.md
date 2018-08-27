# edgeThinning
Zhang-Suen thinning algorithm using OpenCV（C++)  wrapped by boost-python for python3.x or python2.x calls.
# Introduction
The algorithm is explained in ["A fast parallel algorithm for thinning digital patterns"](http://www-prima.inrialpes.fr/perso/Tran/Draft/gateway.cfm.pdf) by T.Y. Zhang and C.Y. Suen. This project is based on the work of [bsdnoobz](https://github.com/bsdnoobz/zhang-suen-thinning) and [Algomorph](https://github.com/Algomorph/pyboostcvconverter). The purpose of the project is due to the python version of the thinning algorithm is too slow, and the C++ version is not flexible enough. In addition, since the [Weave](https://github.com/scipy/weave) library does not provide support for python3.x, So established such a flexible and speed project.
# Compiling
+ Install Boost library ```sudo apt-get install libboost-all-dev ```  
+ Install cmake  ```sudo apt-get install cmake cmake-gui ```
+ Run CMake and/or CMake-gui with the git repository as the source and a build folder of your choice (in-source builds supported.) Choose desired generator, configure, and generate. Remember to set PYTHON_DESIRED_VERSION to 2.X for python 2 and 3.X for python 3.
+ Build (run make on *nix systems with gcc/eclipse CDT generator from within the build folder)
+ On *nix systems, make install run with root privileges will install the compiled library file. Alternatively, you can manually copy it to the pythonXX/dist-packages directory (replace XX with desired python version).
+ Run python interpreter of your choice, issue
# Example
```
import cv2
import pbcvt
src = cv2.imread('image.png')
cv2.imshow("input", src)
thinned_img = pbcvt.thinning(src)
cv2.imshow('thinning', thinned_img)
cv2.waitKey()
```
