# Autoquilt

An implementation of [Whole-Cloth Quilting Patterns from Photographs](https://textiles-lab.github.io/publications/2017-autoquilt/) by Liu, Hodgins, and McCann.

## Requirements

* CMake
* OpenCV
* CGAL

## Build Instructions (Linux)

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Usage
```
./autoquilt [image] [FDoG image from flowabs] [output pattern file]
```

As a temporary workaround until FDoG is implemented, get the FDoG image from https://github.com/MzHub/flowabs (binaries at https://code.google.com/archive/p/flowabs/downloads).

## TODO

* Segmentation with labeled images and per-label FDoG parameters
* FDoG for required edge generation
* Landmark selection for landmark distance estimation is currently random:
use a better selection algorithm
* Matching edge computation should use an (implicit) line graph so we can add
a curvature term to the cost
* General cleanup
* Test building on other platforms
* OpenMP?
* OpenGL visualizer?
