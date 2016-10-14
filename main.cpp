#include <iostream>
#include <tuple>
#include <boost/version.hpp>
#include <sys/types.h>
#include <stdlib.h>
#include "/usr/include/valgrind/callgrind.h"
#include <stdio.h>
#include <string.h>
#include <sys/sysinfo.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <set>
#include "KDTreeVectorOfVectorsAdaptor.h"

#define MEASUREMENTS 600
#define CUT_MEASUREMENTS 100
#define GB 1000000000

using std::cout;
using std::vector;
using std::string;
using std::endl;

//using KDTreeVectorOfVectorsAdaptor



uint32_t getDimensionality(long &num_of_points, float &file_size, string filename) {
    /*figure out the size of the file*/
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    file_size= in.tellg();

    std::ifstream in_file_stream(filename);
    uint32_t dimensionality=0;
    if (!in_file_stream) {
        cout << "failed to read input" << endl;
        exit(-1);
    }
    std::string line;

    bool first_point = true;
    while (getline(in_file_stream, line)) {

        //ignore comment lines
        if (line[0] == '#') {
            continue;
        } else {

            if (first_point) {
                std::string word;
                //istringstream makes the line string an input string stream
                std::istringstream record(line);
                vector<std::string> tokens;
                /*a record contains a number of coordinates and a cluster id:
                    2.58119273751133 -3.0897997256242977 1*/
                while (record >> word) {
                    tokens.push_back(word);
                }

                dimensionality = tokens.size()-1;
                tokens.clear();
            }
            //count the points in the file
            ++num_of_points;
            first_point = false;
        }


    }


    std::cout << "num of points:" << num_of_points << std::endl;
    return dimensionality;

}

std::set<uint32_t> getRandomIndices(long const num_of_points) {
    std::set<uint32_t> random_indices;
    srand(time(NULL));
    while (random_indices.size() != MEASUREMENTS) {
        int randNum = rand() % (num_of_points);
        random_indices.insert(randNum);
    }

    return random_indices;
}



void calculate_statistics(vector<double> &times, float &file_size) {

    //sort times
    std::sort(times.begin(), times.end());

    times.resize(MEASUREMENTS - CUT_MEASUREMENTS);

    /*estimated mean*/
    double sum = std::accumulate(times.cbegin(), times.cend(), 0.0);
    double mean = sum / times.size();

    vector<double> times_minus_mean;

    /*subtract mean from every vector element*/
    std::transform(times.cbegin(), times.cend(), std::back_inserter(times_minus_mean),
                   [&mean](double const x) { return x - mean; });

    double innerProduct = std::inner_product(times_minus_mean.cbegin(), times_minus_mean.cend(),
                                             times_minus_mean.cbegin(), 0.0);

    double std = std::sqrt(innerProduct / times_minus_mean.size());

    double std_error = std / std::sqrt((MEASUREMENTS - CUT_MEASUREMENTS));


    cout << "#Dataset size|Time|Standard error" << endl;
    cout << "bst:" << file_size / GB << "\t" << mean << "\t" << std_error << "\t" << endl;

}

void readFile(vector<vector<double>> &all_points, string filename) {


    cout<<"reading file now"<<endl;
    std::ifstream in_file_stream(filename);
    if (!in_file_stream) {
        cout << "failed to read input" << endl;
        exit(-1);
    }
    string line, word;
    vector<string> tokens;
    while (getline(in_file_stream, line)) {

        //ignore comment lines
        if (line[0] == '#') {
            continue;
        }
        //istringstream makes the line string an input string stream
        std::istringstream record(line);

        /*a record contains a number of coordinates and a cluster id:
        2.58119273751133 -3.0897997256242977 1*/
        while (record >> word) {
            tokens.push_back(word);
        }

        vector<double > coordinates;
        double one_dimension;
        for (string record:tokens) {
            one_dimension = std::stod(record);
            coordinates.push_back(one_dimension);
        }

        all_points.push_back(coordinates);
//        cluster_id = static_cast<long>(coordinates.back());
//    cluster_id=static_cast<long>(id_generator);
//        coordinates.pop_back();


        tokens.clear();

    }
    cout<<"reading file done"<<endl;

}


int main(int argc, char **argv) {

    float file_size=0;
    long num_of_points=0;
    vector<double> times;
    vector<vector<double>> kdtree_points ;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;


    int dimensionality = getDimensionality(num_of_points,file_size,argv[argc-1]);

    cout<<"Dimensionality is:"<<dimensionality<<endl;
    cout<<"Num of points:"<<num_of_points<<endl;



     readFile(kdtree_points, argv[argc - 1]);
//    KDTreeVectorOfVectorsAdaptor kdtree(dimensionality, const VectorOfVectorsType &mat, const int leaf_max_size = 10);
    typedef KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<double> >, double> my_kd_tree_t;

    my_kd_tree_t mat_index(dimensionality /*dim*/, kdtree_points, 10 /* max leaf */ );
    mat_index.index->buildIndex();
    cout << "KD TREE SIZE:" << mat_index.index->size() << endl;


    auto random_indices=getRandomIndices(num_of_points);

    for (auto index:random_indices){
        //index is just a number to index the vector of vectors
        cout << "looking for point:" << index << endl;
        auto coordinates=kdtree_points.at(index);
        std::for_each(coordinates.cbegin(),coordinates.cend(),[](double d){cout<<d<<",";});
        cout<<endl;

        t1 = std::chrono::high_resolution_clock::now();

        const size_t num_results = 1;
        std::vector<size_t> ret_indexes(num_results);
        std::vector<double> out_dists_sqr(num_results);



        nanoflann::KNNResultSet<double> resultSet(num_results);


        resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
        mat_index.index->findNeighbors(resultSet, &coordinates[0], nanoflann::SearchParams(10));
//        for (double d:out_dists_sqr){
//            cout<<"distance:"<<d<<endl;
//        }
//
//        for (auto d:ret_indexes){
//            cout<<"retindex:"<<d<<endl;
//        }
        long cluster_id = mat_index.m_data[ret_indexes[0]][mat_index.m_data[ret_indexes[0]].size() - 1];


        t2 = std::chrono::high_resolution_clock::now();
        cout<<"cluster_id:"<<cluster_id<<endl;

//        if (cluster_id!=point.getCluster_id()){
//            cout<<"Wrong Cluster id retrieved"<<endl;
//            exit(14);
//        }

        double slib_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        times.push_back(slib_time);
    }


    calculate_statistics(times,file_size);
    exit(111);

}

