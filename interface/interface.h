#pragma once

#include <memory>
#include <vector>
#include <unistd.h>
#include <iostream>
// #include <Transform.hpp>

// Faster debug without true import of Transform
enum class TransformType {
    crop,
    resize,
    binaryScale,
    inversion,
    equalize,
    restriction,
    binaryScaleByMedian,
  };


typedef struct {
    std::string dataset_path;
    std::string output_path;
    unsigned rescale_size;
    std::vector<TransformType> preprocess_transformations;
    std::vector<TransformType> postprocess_transformations;
    

} parameters_t;

static void printParameters(const parameters_t& parameters) {
    std::cout << "==== PARAMETERS ====" << std::endl;
    
    std::cout << "dataset_path = " << parameters.dataset_path << std::endl;
    std::cout << "output_path = " << parameters.output_path << std::endl;
    std::cout << "rescale_size = " << parameters.rescale_size << std::endl;
    //std::cout << "preprocess_transformations = " << [] << std::endl;
    //std::cout << "postprocess_transformations = " << [] << std::endl;
    
    std::cout << "====================" << std::endl;
}

static PyObject* INTERFACE_createAndTrain(PyObject *self, PyObject *args);