# nanoflann-kdtree-example
A simple example that uses nanoflann library to load points into a KD-Tree. The nice thing about FLANN and nanoflann is that it can support an arbitrary number of dimensions. So if you need a dimension agnostic KD-Tree nanoflann (and FLANN) is ideal

## The application does the following:
1. Reads the points from a file and inserts them directly into a KD-Tree - no intermediate structure is used as this can easily fill up your memory when dealing with big data. I have used **valgrind massif** to verify that memory usage is almost equal (the adaptors allocate some extra memory) to the size of the input
2. It stores a cluster id along with each point
3. **Nearest Neighbour search is done on the d-1 first positions of each vector** (which is the point). The **d<sup>th</sup> position of each vector is the cluster id** to which the point belongs. To achieve this behaviour KDTreeVectorOfVectorsAdaptor:63 was modified to KDTreeVectorOfVectorsAdaptor:64.  
4. It selects 600 random points and uses them as query points 
5. It queries the KD-Tree and estimates the average time for retrieving a point. It also estimates the error of the measurements 

## Notes:
1. Requires cmake version 3.6 - although it can probably work older versions 
2. A **sample file** is provided

## How to run on Linux
```bash
git clone https://github.com/gevago01/nanoflann-kdtree-example
mkdir build
cd build
cmake ..
make
./SpatialIndexRtree ../sample.txt
```
