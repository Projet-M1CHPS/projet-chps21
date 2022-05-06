#include "NeuralNetwork.hpp"
#include <gtest/gtest.h>


#include "ActivationFunction.hpp"
#include "CNN.hpp"
#include "CNNLayer.hpp"
#include "CNNTopology.hpp"
#include "CNNTopologyLayer.hpp"


using namespace nnet;
using namespace math;


TEST(CNNTopologyTest, canCreateTopologyLayerConvolution) {
  CNNTopologyLayerConvolution topology_layer({6, 6}, 2, {2, 2}, af::ActivationFunctionType::relu,
                                             2);

  ASSERT_EQ(5, topology_layer.getOutputSize().first);
  ASSERT_EQ(5, topology_layer.getOutputSize().second);
  ASSERT_EQ(2, topology_layer.getFeatures());
  ASSERT_EQ(2, topology_layer.getFilterSize().first);
  ASSERT_EQ(2, topology_layer.getFilterSize().second);
}

TEST(CNNTopologyTest, canCreateTopologyLayerAvgPooling) {
  CNNTopologyLayerAvgPooling topology_layer({6, 6}, {2, 2}, 2);

  ASSERT_EQ(5, topology_layer.getOutputSize().first);
  ASSERT_EQ(5, topology_layer.getOutputSize().second);
  ASSERT_EQ(1, topology_layer.getFeatures());
  ASSERT_EQ(2, topology_layer.getFilterSize().first);
  ASSERT_EQ(2, topology_layer.getFilterSize().second);
}

TEST(CNNTopologyTest, canCreateTopology) {
  std::string str_topology("6 6 relu convolution 2 2 2 convolution 2 2 2 pooling max 2 2");
  auto topology = stringToTopology(str_topology);

  ASSERT_EQ(af::ActivationFunctionType::relu, topology.getActivationFunction());
  ASSERT_EQ(6, topology.getInputSize().first);
  ASSERT_EQ(6, topology.getInputSize().second);
  ASSERT_EQ(3, topology.getDepth());
  ASSERT_EQ(4, topology.getNBranchFinal());
  ASSERT_EQ(36, topology.getCNNOutputSize());
}

TEST(CNNTopologyTest, throwInvalidTopology) {
  std::string str_topology("6 6 relu conv 2 2 2 convolution 2 2 2 pooling max 2 2");
  ASSERT_ANY_THROW(stringToTopology(str_topology));
}


TEST(CNNLayerTest, canCreateConvolutionLayer) {
  CNNConvolutionLayer layer({5, 5}, {2, 2}, 2, af::ActivationFunctionType::relu, 1);

  ASSERT_EQ(true, layer.hasWeight());
  ASSERT_EQ(5, layer.getOutputSize().first);
  ASSERT_EQ(5, layer.getOutputSize().second);

  auto &filter = layer.getFilter();
  ASSERT_EQ(2, filter.getRows());
  ASSERT_EQ(2, filter.getCols());
  ASSERT_EQ(2, filter.getDepth());
}

TEST(CNNLayerTest, canCopyConvolutionLayer) {
  CNNConvolutionLayer layer({5, 5}, {2, 2}, 2, af::ActivationFunctionType::relu, 1);

  CNNConvolutionLayer copy(layer);

  ASSERT_EQ(copy.hasWeight(), layer.hasWeight());
  ASSERT_EQ(copy.getOutputSize().first, layer.getOutputSize().first);
  ASSERT_EQ(copy.getOutputSize().second, layer.getOutputSize().second);

  auto &filter = layer.getFilter();
  auto &copy_filter = copy.getFilter();
  ASSERT_EQ(copy_filter.getRows(), filter.getRows());
  ASSERT_EQ(copy_filter.getCols(), filter.getCols());
  ASSERT_EQ(copy_filter.getDepth(), filter.getDepth());
}

TEST(CNNLayerTest, canComputeConvolutionLayer) {
  auto &queue = utils::cl_wrapper.getDefaultQueue();
  CNNConvolutionLayer layer({5, 5}, {2, 2}, 2, af::ActivationFunctionType::relu, 2);

  FloatMatrix f1(2, 2);
  {
    f1(0, 0) = 2.f;
    f1(0, 1) = 1.f;
    f1(1, 0) = 0.5f;
    f1(1, 1) = 1.5f;
  }
  FloatMatrix f2(2, 2);
  {
    f2(0, 0) = 1.f;
    f2(0, 1) = 1.f;
    f2(1, 0) = 1.f;
    f2(1, 1) = 1.f;
  }
  FloatMatrix f3(2, 2);
  {
    f3(0, 0) = 2.f;
    f3(0, 1) = 2.f;
    f3(1, 0) = 2.f;
    f3(1, 1) = 2.f;
  }
  FloatMatrix f4(2, 2);
  {
    f4(0, 0) = 4.f;
    f4(0, 1) = 4.f;
    f4(1, 0) = 4.f;
    f4(1, 1) = 4.f;
  }
  math::FloatMatrix input1(6, 6);
  {
    input1(0, 0) = 1.f;
    input1(0, 1) = 2.f;
    input1(0, 2) = 1.f;
    input1(0, 3) = 1.f;
    input1(0, 4) = 4.f;
    input1(0, 5) = 1.f;
    input1(1, 0) = 2.f;
    input1(1, 1) = 1.f;
    input1(1, 2) = 1.f;
    input1(1, 3) = 2.f;
    input1(1, 4) = 2.f;
    input1(1, 5) = 1.f;
    input1(2, 0) = 4.f;
    input1(2, 1) = 3.f;
    input1(2, 2) = 2.f;
    input1(2, 3) = 1.f;
    input1(2, 4) = 2.f;
    input1(2, 5) = 1.f;
    input1(3, 0) = 1.f;
    input1(3, 1) = 5.f;
    input1(3, 2) = 1.f;
    input1(3, 3) = 1.f;
    input1(3, 4) = 2.f;
    input1(3, 5) = 1.f;
    input1(4, 0) = 2.f;
    input1(4, 1) = 1.f;
    input1(4, 2) = 1.f;
    input1(4, 3) = 4.f;
    input1(4, 4) = 1.f;
    input1(4, 5) = 1.f;
    input1(5, 0) = 2.f;
    input1(5, 1) = 1.f;
    input1(5, 2) = 4.f;
    input1(5, 3) = 2.f;
    input1(5, 4) = 4.f;
    input1(5, 5) = 1.f;
  }
  math::FloatMatrix input2(input1);
  input2(0, 0) = 100.f;

  auto &filter = layer.getFilter();
  filter[0] = f1;
  filter[1] = f2;
  filter[2] = f3;
  filter[3] = f4;

  clFTensor input_tensor(6, 6, 4);
  input_tensor[0] = input1;
  input_tensor[1] = input2;
  input_tensor[2] = input1;
  input_tensor[3] = input2;

  clFTensor output_tensor = layer.compute(queue, input_tensor);

  queue.finish();

  std::array<FloatMatrix, 8> valid_matrix;
  for (auto &matrix : valid_matrix) matrix = FloatMatrix(5, 5);

  {
    valid_matrix[0](0, 0) = 6.5f;
    valid_matrix[0](0, 1) = 7.f;
    valid_matrix[0](0, 2) = 6.5f;
    valid_matrix[0](0, 3) = 10.f;
    valid_matrix[0](0, 4) = 11.5f;
    valid_matrix[0](1, 0) = 11.5f;
    valid_matrix[0](1, 1) = 7.5f;
    valid_matrix[0](1, 2) = 6.5f;
    valid_matrix[0](1, 3) = 9.5f;
    valid_matrix[0](1, 4) = 7.5f;
    valid_matrix[0](2, 0) = 19.f;
    valid_matrix[0](2, 1) = 12.f;
    valid_matrix[0](2, 2) = 7.f;
    valid_matrix[0](2, 3) = 7.5f;
    valid_matrix[0](2, 4) = 7.5f;
    valid_matrix[0](3, 0) = 9.5f;
    valid_matrix[0](3, 1) = 13.f;
    valid_matrix[0](3, 2) = 9.5f;
    valid_matrix[0](3, 3) = 7.5f;
    valid_matrix[0](3, 4) = 7.f;
    valid_matrix[0](4, 0) = 7.5f;
    valid_matrix[0](4, 1) = 9.5f;
    valid_matrix[0](4, 2) = 11.f;
    valid_matrix[0](4, 3) = 16.f;
    valid_matrix[0](4, 4) = 6.5f;

    valid_matrix[1](0, 0) = 204.5f;
    valid_matrix[1](0, 1) = 7.f;
    valid_matrix[1](0, 2) = 6.5f;
    valid_matrix[1](0, 3) = 10.f;
    valid_matrix[1](0, 4) = 11.5f;
    valid_matrix[1](1, 0) = 11.5f;
    valid_matrix[1](1, 1) = 7.5f;
    valid_matrix[1](1, 2) = 6.5f;
    valid_matrix[1](1, 3) = 9.5f;
    valid_matrix[1](1, 4) = 7.5f;
    valid_matrix[1](2, 0) = 19.f;
    valid_matrix[1](2, 1) = 12.f;
    valid_matrix[1](2, 2) = 7.f;
    valid_matrix[1](2, 3) = 7.5f;
    valid_matrix[1](2, 4) = 7.5f;
    valid_matrix[1](3, 0) = 9.5f;
    valid_matrix[1](3, 1) = 13.f;
    valid_matrix[1](3, 2) = 9.5f;
    valid_matrix[1](3, 3) = 7.5f;
    valid_matrix[1](3, 4) = 7.f;
    valid_matrix[1](4, 0) = 7.5f;
    valid_matrix[1](4, 1) = 9.5f;
    valid_matrix[1](4, 2) = 11.f;
    valid_matrix[1](4, 3) = 16.f;
    valid_matrix[1](4, 4) = 6.5f;

    valid_matrix[2](0, 0) = 6.f;
    valid_matrix[2](0, 1) = 5.f;
    valid_matrix[2](0, 2) = 5.f;
    valid_matrix[2](0, 3) = 9.f;
    valid_matrix[2](0, 4) = 8.f;
    valid_matrix[2](1, 0) = 10.f;
    valid_matrix[2](1, 1) = 7.f;
    valid_matrix[2](1, 2) = 6.f;
    valid_matrix[2](1, 3) = 7.f;
    valid_matrix[2](1, 4) = 6.f;
    valid_matrix[2](2, 0) = 13.f;
    valid_matrix[2](2, 1) = 11.f;
    valid_matrix[2](2, 2) = 5.f;
    valid_matrix[2](2, 3) = 6.f;
    valid_matrix[2](2, 4) = 6.f;
    valid_matrix[2](3, 0) = 9.f;
    valid_matrix[2](3, 1) = 8.f;
    valid_matrix[2](3, 2) = 7.f;
    valid_matrix[2](3, 3) = 8.f;
    valid_matrix[2](3, 4) = 5.f;
    valid_matrix[2](4, 0) = 6.f;
    valid_matrix[2](4, 1) = 7.f;
    valid_matrix[2](4, 2) = 11.f;
    valid_matrix[2](4, 3) = 11.f;
    valid_matrix[2](4, 4) = 7.f;

    valid_matrix[3](0, 0) = 105.f;
    valid_matrix[3](0, 1) = 5.f;
    valid_matrix[3](0, 2) = 5.f;
    valid_matrix[3](0, 3) = 9.f;
    valid_matrix[3](0, 4) = 8.f;
    valid_matrix[3](1, 0) = 10.f;
    valid_matrix[3](1, 1) = 7.f;
    valid_matrix[3](1, 2) = 6.f;
    valid_matrix[3](1, 3) = 7.f;
    valid_matrix[3](1, 4) = 6.f;
    valid_matrix[3](2, 0) = 13.f;
    valid_matrix[3](2, 1) = 11.f;
    valid_matrix[3](2, 2) = 5.f;
    valid_matrix[3](2, 3) = 6.f;
    valid_matrix[3](2, 4) = 6.f;
    valid_matrix[3](3, 0) = 9.f;
    valid_matrix[3](3, 1) = 8.f;
    valid_matrix[3](3, 2) = 7.f;
    valid_matrix[3](3, 3) = 8.f;
    valid_matrix[3](3, 4) = 5.f;
    valid_matrix[3](4, 0) = 6.f;
    valid_matrix[3](4, 1) = 7.f;
    valid_matrix[3](4, 2) = 11.f;
    valid_matrix[3](4, 3) = 11.f;
    valid_matrix[3](4, 4) = 7.f;

    valid_matrix[4](0, 0) = 12.f;
    valid_matrix[4](0, 1) = 10.f;
    valid_matrix[4](0, 2) = 10.f;
    valid_matrix[4](0, 3) = 18.f;
    valid_matrix[4](0, 4) = 16.f;
    valid_matrix[4](1, 0) = 20.f;
    valid_matrix[4](1, 1) = 14.f;
    valid_matrix[4](1, 2) = 12.f;
    valid_matrix[4](1, 3) = 14.f;
    valid_matrix[4](1, 4) = 12.f;
    valid_matrix[4](2, 0) = 26.f;
    valid_matrix[4](2, 1) = 22.f;
    valid_matrix[4](2, 2) = 10.f;
    valid_matrix[4](2, 3) = 12.f;
    valid_matrix[4](2, 4) = 12.f;
    valid_matrix[4](3, 0) = 18.f;
    valid_matrix[4](3, 1) = 16.f;
    valid_matrix[4](3, 2) = 14.f;
    valid_matrix[4](3, 3) = 16.f;
    valid_matrix[4](3, 4) = 10.f;
    valid_matrix[4](4, 0) = 12.f;
    valid_matrix[4](4, 1) = 14.f;
    valid_matrix[4](4, 2) = 22.f;
    valid_matrix[4](4, 3) = 22.f;
    valid_matrix[4](4, 4) = 14.f;

    valid_matrix[5](0, 0) = 210.f;
    valid_matrix[5](0, 1) = 10.f;
    valid_matrix[5](0, 2) = 10.f;
    valid_matrix[5](0, 3) = 18.f;
    valid_matrix[5](0, 4) = 16.f;
    valid_matrix[5](1, 0) = 20.f;
    valid_matrix[5](1, 1) = 14.f;
    valid_matrix[5](1, 2) = 12.f;
    valid_matrix[5](1, 3) = 14.f;
    valid_matrix[5](1, 4) = 12.f;
    valid_matrix[5](2, 0) = 26.f;
    valid_matrix[5](2, 1) = 22.f;
    valid_matrix[5](2, 2) = 10.f;
    valid_matrix[5](2, 3) = 12.f;
    valid_matrix[5](2, 4) = 12.f;
    valid_matrix[5](3, 0) = 18.f;
    valid_matrix[5](3, 1) = 16.f;
    valid_matrix[5](3, 2) = 14.f;
    valid_matrix[5](3, 3) = 16.f;
    valid_matrix[5](3, 4) = 10.f;
    valid_matrix[5](4, 0) = 12.f;
    valid_matrix[5](4, 1) = 14.f;
    valid_matrix[5](4, 2) = 22.f;
    valid_matrix[5](4, 3) = 22.f;
    valid_matrix[5](4, 4) = 14.f;

    valid_matrix[6](0, 0) = 24.f;
    valid_matrix[6](0, 1) = 20.f;
    valid_matrix[6](0, 2) = 20.f;
    valid_matrix[6](0, 3) = 36.f;
    valid_matrix[6](0, 4) = 32.f;
    valid_matrix[6](1, 0) = 40.f;
    valid_matrix[6](1, 1) = 28.f;
    valid_matrix[6](1, 2) = 24.f;
    valid_matrix[6](1, 3) = 28.f;
    valid_matrix[6](1, 4) = 24.f;
    valid_matrix[6](2, 0) = 52.f;
    valid_matrix[6](2, 1) = 44.f;
    valid_matrix[6](2, 2) = 20.f;
    valid_matrix[6](2, 3) = 24.f;
    valid_matrix[6](2, 4) = 24.f;
    valid_matrix[6](3, 0) = 36.f;
    valid_matrix[6](3, 1) = 32.f;
    valid_matrix[6](3, 2) = 28.f;
    valid_matrix[6](3, 3) = 32.f;
    valid_matrix[6](3, 4) = 20.f;
    valid_matrix[6](4, 0) = 24.f;
    valid_matrix[6](4, 1) = 28.f;
    valid_matrix[6](4, 2) = 44.f;
    valid_matrix[6](4, 3) = 44.f;
    valid_matrix[6](4, 4) = 28.f;

    valid_matrix[7](0, 0) = 420.f;
    valid_matrix[7](0, 1) = 20.f;
    valid_matrix[7](0, 2) = 20.f;
    valid_matrix[7](0, 3) = 36.f;
    valid_matrix[7](0, 4) = 32.f;
    valid_matrix[7](1, 0) = 40.f;
    valid_matrix[7](1, 1) = 28.f;
    valid_matrix[7](1, 2) = 24.f;
    valid_matrix[7](1, 3) = 28.f;
    valid_matrix[7](1, 4) = 24.f;
    valid_matrix[7](2, 0) = 52.f;
    valid_matrix[7](2, 1) = 44.f;
    valid_matrix[7](2, 2) = 20.f;
    valid_matrix[7](2, 3) = 24.f;
    valid_matrix[7](2, 4) = 24.f;
    valid_matrix[7](3, 0) = 36.f;
    valid_matrix[7](3, 1) = 32.f;
    valid_matrix[7](3, 2) = 28.f;
    valid_matrix[7](3, 3) = 32.f;
    valid_matrix[7](3, 4) = 20.f;
    valid_matrix[7](4, 0) = 24.f;
    valid_matrix[7](4, 1) = 28.f;
    valid_matrix[7](4, 2) = 44.f;
    valid_matrix[7](4, 3) = 44.f;
    valid_matrix[7](4, 4) = 28.f;
  }
  for (size_t ii = 1; ii < 2; ii++) {
    for (size_t i = 0; i < 5; i++) {
      for (size_t j = 0; j < 5; j++) {
        EXPECT_NEAR(valid_matrix[ii](i, j), output_tensor[ii].toFloatMatrix(true)(i, j), 0.1f);
      }
    }
  }
}

TEST(CNNLayerTest, canComputeBPConvolutionLayer) {
  auto &queue = utils::cl_wrapper.getDefaultQueue();
  nnet::CNNConvolutionLayer layer({5, 5}, {2, 2}, 2, af::ActivationFunctionType::relu, 2);
  nnet::CNNStorageBPConvolution storage;

  FloatMatrix f1(2, 2);
  {
    f1(0, 0) = 2.f;
    f1(0, 1) = 1.f;
    f1(1, 0) = 0.5f;
    f1(1, 1) = 1.5f;
  }
  FloatMatrix f2(2, 2);
  {
    f2(0, 0) = 1.f;
    f2(0, 1) = 1.f;
    f2(1, 0) = 1.f;
    f2(1, 1) = 1.f;
  }
  FloatMatrix f3(2, 2);
  {
    f3(0, 0) = 2.f;
    f3(0, 1) = 2.f;
    f3(1, 0) = 2.f;
    f3(1, 1) = 2.f;
  }
  FloatMatrix f4(2, 2);
  {
    f4(0, 0) = 4.f;
    f4(0, 1) = 4.f;
    f4(1, 0) = 4.f;
    f4(1, 1) = 4.f;
  }
  auto &filter = layer.getFilter();
  filter[0] = f1;
  filter[1] = f2;
  filter[2] = f3;
  filter[3] = f4;

  std::array<math::FloatMatrix, 2> input;
  for (auto &matrix : input) matrix = FloatMatrix(6, 6);
  {
    input[0](0, 0) = 1.f;
    input[1](0, 0) = 100.f;
    input[0](0, 1) = input[1](0, 1) = 2.f;
    input[0](0, 2) = input[1](0, 2) = 1.f;
    input[0](0, 3) = input[1](0, 3) = 1.f;
    input[0](0, 4) = input[1](0, 4) = 4.f;
    input[0](0, 5) = input[1](0, 5) = 1.f;
    input[0](1, 0) = input[1](1, 0) = 2.f;
    input[0](1, 1) = input[1](1, 1) = 1.f;
    input[0](1, 2) = input[1](1, 2) = 1.f;
    input[0](1, 3) = input[1](1, 3) = 2.f;
    input[0](1, 4) = input[1](1, 4) = 2.f;
    input[0](1, 5) = input[1](1, 5) = 1.f;
    input[0](2, 0) = input[1](2, 0) = 4.f;
    input[0](2, 1) = input[1](2, 1) = 3.f;
    input[0](2, 2) = input[1](2, 2) = 2.f;
    input[0](2, 3) = input[1](2, 3) = 1.f;
    input[0](2, 4) = input[1](2, 4) = 2.f;
    input[0](2, 5) = input[1](2, 5) = 1.f;
    input[0](3, 0) = input[1](3, 0) = 1.f;
    input[0](3, 1) = input[1](3, 1) = 5.f;
    input[0](3, 2) = input[1](3, 2) = 1.f;
    input[0](3, 3) = input[1](3, 3) = 1.f;
    input[0](3, 4) = input[1](3, 4) = 2.f;
    input[0](3, 5) = input[1](3, 5) = 1.f;
    input[0](4, 0) = input[1](4, 0) = 2.f;
    input[0](4, 1) = input[1](4, 1) = 1.f;
    input[0](4, 2) = input[1](4, 2) = 1.f;
    input[0](4, 3) = input[1](4, 3) = 4.f;
    input[0](4, 4) = input[1](4, 4) = 1.f;
    input[0](4, 5) = input[1](4, 5) = 1.f;
    input[0](5, 0) = input[1](5, 0) = 2.f;
    input[0](5, 1) = input[1](5, 1) = 1.f;
    input[0](5, 2) = input[1](5, 2) = 4.f;
    input[0](5, 3) = input[1](5, 3) = 2.f;
    input[0](5, 4) = input[1](5, 4) = 4.f;
    input[0](5, 5) = input[1](5, 5) = 1.f;
  }

  clFTensor input_tensor(6, 6, 4);
  input_tensor[0] = input[0];
  input_tensor[1] = input[1];
  input_tensor[2] = input[0];
  input_tensor[3] = input[1];

  clFTensor output_tensor = layer.computeForward(queue, input_tensor, storage);

  queue.finish();

  std::array<FloatMatrix, 8> valid_matrix_output;
  for (auto &matrix : valid_matrix_output) matrix = FloatMatrix(5, 5);
  {
    valid_matrix_output[0](0, 0) = 6.5f;
    valid_matrix_output[0](0, 1) = 7.f;
    valid_matrix_output[0](0, 2) = 6.5f;
    valid_matrix_output[0](0, 3) = 10.f;
    valid_matrix_output[0](0, 4) = 11.5f;
    valid_matrix_output[0](1, 0) = 11.5f;
    valid_matrix_output[0](1, 1) = 7.5f;
    valid_matrix_output[0](1, 2) = 6.5f;
    valid_matrix_output[0](1, 3) = 9.5f;
    valid_matrix_output[0](1, 4) = 7.5f;
    valid_matrix_output[0](2, 0) = 19.f;
    valid_matrix_output[0](2, 1) = 12.f;
    valid_matrix_output[0](2, 2) = 7.f;
    valid_matrix_output[0](2, 3) = 7.5f;
    valid_matrix_output[0](2, 4) = 7.5f;
    valid_matrix_output[0](3, 0) = 9.5f;
    valid_matrix_output[0](3, 1) = 13.f;
    valid_matrix_output[0](3, 2) = 9.5f;
    valid_matrix_output[0](3, 3) = 7.5f;
    valid_matrix_output[0](3, 4) = 7.f;
    valid_matrix_output[0](4, 0) = 7.5f;
    valid_matrix_output[0](4, 1) = 9.5f;
    valid_matrix_output[0](4, 2) = 11.f;
    valid_matrix_output[0](4, 3) = 16.f;
    valid_matrix_output[0](4, 4) = 6.5f;

    valid_matrix_output[1](0, 0) = 204.5f;
    valid_matrix_output[1](0, 1) = 7.f;
    valid_matrix_output[1](0, 2) = 6.5f;
    valid_matrix_output[1](0, 3) = 10.f;
    valid_matrix_output[1](0, 4) = 11.5f;
    valid_matrix_output[1](1, 0) = 11.5f;
    valid_matrix_output[1](1, 1) = 7.5f;
    valid_matrix_output[1](1, 2) = 6.5f;
    valid_matrix_output[1](1, 3) = 9.5f;
    valid_matrix_output[1](1, 4) = 7.5f;
    valid_matrix_output[1](2, 0) = 19.f;
    valid_matrix_output[1](2, 1) = 12.f;
    valid_matrix_output[1](2, 2) = 7.f;
    valid_matrix_output[1](2, 3) = 7.5f;
    valid_matrix_output[1](2, 4) = 7.5f;
    valid_matrix_output[1](3, 0) = 9.5f;
    valid_matrix_output[1](3, 1) = 13.f;
    valid_matrix_output[1](3, 2) = 9.5f;
    valid_matrix_output[1](3, 3) = 7.5f;
    valid_matrix_output[1](3, 4) = 7.f;
    valid_matrix_output[1](4, 0) = 7.5f;
    valid_matrix_output[1](4, 1) = 9.5f;
    valid_matrix_output[1](4, 2) = 11.f;
    valid_matrix_output[1](4, 3) = 16.f;
    valid_matrix_output[1](4, 4) = 6.5f;

    valid_matrix_output[2](0, 0) = 6.f;
    valid_matrix_output[2](0, 1) = 5.f;
    valid_matrix_output[2](0, 2) = 5.f;
    valid_matrix_output[2](0, 3) = 9.f;
    valid_matrix_output[2](0, 4) = 8.f;
    valid_matrix_output[2](1, 0) = 10.f;
    valid_matrix_output[2](1, 1) = 7.f;
    valid_matrix_output[2](1, 2) = 6.f;
    valid_matrix_output[2](1, 3) = 7.f;
    valid_matrix_output[2](1, 4) = 6.f;
    valid_matrix_output[2](2, 0) = 13.f;
    valid_matrix_output[2](2, 1) = 11.f;
    valid_matrix_output[2](2, 2) = 5.f;
    valid_matrix_output[2](2, 3) = 6.f;
    valid_matrix_output[2](2, 4) = 6.f;
    valid_matrix_output[2](3, 0) = 9.f;
    valid_matrix_output[2](3, 1) = 8.f;
    valid_matrix_output[2](3, 2) = 7.f;
    valid_matrix_output[2](3, 3) = 8.f;
    valid_matrix_output[2](3, 4) = 5.f;
    valid_matrix_output[2](4, 0) = 6.f;
    valid_matrix_output[2](4, 1) = 7.f;
    valid_matrix_output[2](4, 2) = 11.f;
    valid_matrix_output[2](4, 3) = 11.f;
    valid_matrix_output[2](4, 4) = 7.f;

    valid_matrix_output[3](0, 0) = 105.f;
    valid_matrix_output[3](0, 1) = 5.f;
    valid_matrix_output[3](0, 2) = 5.f;
    valid_matrix_output[3](0, 3) = 9.f;
    valid_matrix_output[3](0, 4) = 8.f;
    valid_matrix_output[3](1, 0) = 10.f;
    valid_matrix_output[3](1, 1) = 7.f;
    valid_matrix_output[3](1, 2) = 6.f;
    valid_matrix_output[3](1, 3) = 7.f;
    valid_matrix_output[3](1, 4) = 6.f;
    valid_matrix_output[3](2, 0) = 13.f;
    valid_matrix_output[3](2, 1) = 11.f;
    valid_matrix_output[3](2, 2) = 5.f;
    valid_matrix_output[3](2, 3) = 6.f;
    valid_matrix_output[3](2, 4) = 6.f;
    valid_matrix_output[3](3, 0) = 9.f;
    valid_matrix_output[3](3, 1) = 8.f;
    valid_matrix_output[3](3, 2) = 7.f;
    valid_matrix_output[3](3, 3) = 8.f;
    valid_matrix_output[3](3, 4) = 5.f;
    valid_matrix_output[3](4, 0) = 6.f;
    valid_matrix_output[3](4, 1) = 7.f;
    valid_matrix_output[3](4, 2) = 11.f;
    valid_matrix_output[3](4, 3) = 11.f;
    valid_matrix_output[3](4, 4) = 7.f;

    valid_matrix_output[4](0, 0) = 12.f;
    valid_matrix_output[4](0, 1) = 10.f;
    valid_matrix_output[4](0, 2) = 10.f;
    valid_matrix_output[4](0, 3) = 18.f;
    valid_matrix_output[4](0, 4) = 16.f;
    valid_matrix_output[4](1, 0) = 20.f;
    valid_matrix_output[4](1, 1) = 14.f;
    valid_matrix_output[4](1, 2) = 12.f;
    valid_matrix_output[4](1, 3) = 14.f;
    valid_matrix_output[4](1, 4) = 12.f;
    valid_matrix_output[4](2, 0) = 26.f;
    valid_matrix_output[4](2, 1) = 22.f;
    valid_matrix_output[4](2, 2) = 10.f;
    valid_matrix_output[4](2, 3) = 12.f;
    valid_matrix_output[4](2, 4) = 12.f;
    valid_matrix_output[4](3, 0) = 18.f;
    valid_matrix_output[4](3, 1) = 16.f;
    valid_matrix_output[4](3, 2) = 14.f;
    valid_matrix_output[4](3, 3) = 16.f;
    valid_matrix_output[4](3, 4) = 10.f;
    valid_matrix_output[4](4, 0) = 12.f;
    valid_matrix_output[4](4, 1) = 14.f;
    valid_matrix_output[4](4, 2) = 22.f;
    valid_matrix_output[4](4, 3) = 22.f;
    valid_matrix_output[4](4, 4) = 14.f;

    valid_matrix_output[5](0, 0) = 210.f;
    valid_matrix_output[5](0, 1) = 10.f;
    valid_matrix_output[5](0, 2) = 10.f;
    valid_matrix_output[5](0, 3) = 18.f;
    valid_matrix_output[5](0, 4) = 16.f;
    valid_matrix_output[5](1, 0) = 20.f;
    valid_matrix_output[5](1, 1) = 14.f;
    valid_matrix_output[5](1, 2) = 12.f;
    valid_matrix_output[5](1, 3) = 14.f;
    valid_matrix_output[5](1, 4) = 12.f;
    valid_matrix_output[5](2, 0) = 26.f;
    valid_matrix_output[5](2, 1) = 22.f;
    valid_matrix_output[5](2, 2) = 10.f;
    valid_matrix_output[5](2, 3) = 12.f;
    valid_matrix_output[5](2, 4) = 12.f;
    valid_matrix_output[5](3, 0) = 18.f;
    valid_matrix_output[5](3, 1) = 16.f;
    valid_matrix_output[5](3, 2) = 14.f;
    valid_matrix_output[5](3, 3) = 16.f;
    valid_matrix_output[5](3, 4) = 10.f;
    valid_matrix_output[5](4, 0) = 12.f;
    valid_matrix_output[5](4, 1) = 14.f;
    valid_matrix_output[5](4, 2) = 22.f;
    valid_matrix_output[5](4, 3) = 22.f;
    valid_matrix_output[5](4, 4) = 14.f;

    valid_matrix_output[6](0, 0) = 24.f;
    valid_matrix_output[6](0, 1) = 20.f;
    valid_matrix_output[6](0, 2) = 20.f;
    valid_matrix_output[6](0, 3) = 36.f;
    valid_matrix_output[6](0, 4) = 32.f;
    valid_matrix_output[6](1, 0) = 40.f;
    valid_matrix_output[6](1, 1) = 28.f;
    valid_matrix_output[6](1, 2) = 24.f;
    valid_matrix_output[6](1, 3) = 28.f;
    valid_matrix_output[6](1, 4) = 24.f;
    valid_matrix_output[6](2, 0) = 52.f;
    valid_matrix_output[6](2, 1) = 44.f;
    valid_matrix_output[6](2, 2) = 20.f;
    valid_matrix_output[6](2, 3) = 24.f;
    valid_matrix_output[6](2, 4) = 24.f;
    valid_matrix_output[6](3, 0) = 36.f;
    valid_matrix_output[6](3, 1) = 32.f;
    valid_matrix_output[6](3, 2) = 28.f;
    valid_matrix_output[6](3, 3) = 32.f;
    valid_matrix_output[6](3, 4) = 20.f;
    valid_matrix_output[6](4, 0) = 24.f;
    valid_matrix_output[6](4, 1) = 28.f;
    valid_matrix_output[6](4, 2) = 44.f;
    valid_matrix_output[6](4, 3) = 44.f;
    valid_matrix_output[6](4, 4) = 28.f;

    valid_matrix_output[7](0, 0) = 420.f;
    valid_matrix_output[7](0, 1) = 20.f;
    valid_matrix_output[7](0, 2) = 20.f;
    valid_matrix_output[7](0, 3) = 36.f;
    valid_matrix_output[7](0, 4) = 32.f;
    valid_matrix_output[7](1, 0) = 40.f;
    valid_matrix_output[7](1, 1) = 28.f;
    valid_matrix_output[7](1, 2) = 24.f;
    valid_matrix_output[7](1, 3) = 28.f;
    valid_matrix_output[7](1, 4) = 24.f;
    valid_matrix_output[7](2, 0) = 52.f;
    valid_matrix_output[7](2, 1) = 44.f;
    valid_matrix_output[7](2, 2) = 20.f;
    valid_matrix_output[7](2, 3) = 24.f;
    valid_matrix_output[7](2, 4) = 24.f;
    valid_matrix_output[7](3, 0) = 36.f;
    valid_matrix_output[7](3, 1) = 32.f;
    valid_matrix_output[7](3, 2) = 28.f;
    valid_matrix_output[7](3, 3) = 32.f;
    valid_matrix_output[7](3, 4) = 20.f;
    valid_matrix_output[7](4, 0) = 24.f;
    valid_matrix_output[7](4, 1) = 28.f;
    valid_matrix_output[7](4, 2) = 44.f;
    valid_matrix_output[7](4, 3) = 44.f;
    valid_matrix_output[7](4, 4) = 28.f;
  }

  for (size_t ii = 1; ii < 2; ii++) {
    for (size_t i = 0; i < 5; i++) {
      for (size_t j = 0; j < 5; j++) {
        EXPECT_NEAR(valid_matrix_output[ii](i, j), output_tensor[ii].toFloatMatrix(true)(i, j),
                    0.1f);
      }
    }
  }

  for (size_t ii = 1; ii < 4; ii++) {
    for (size_t i = 0; i < 6; i++) {
      for (size_t j = 0; j < 6; j++) {
        EXPECT_NEAR(input_tensor[ii].toFloatMatrix(true)(i, j),
                    storage.input[ii].toFloatMatrix(true)(i, j), 0.1f);
      }
    }
  }

  clFTensor errors_tensor(5, 5, 8);
  errors_tensor[0].fill(1.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[1].fill(2.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[2].fill(3.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[3].fill(4.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[4].fill(5.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[5].fill(6.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[6].fill(7.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[7].fill(8.f, utils::cl_wrapper.getDefaultQueue(), true);

  clFTensor errors_input = layer.computeBackward(queue, errors_tensor, storage);

  queue.finish();

  std::array<FloatMatrix, 4> valid_matrix_error_input;
  for (auto &matrix : valid_matrix_error_input) matrix = FloatMatrix(6, 6);
  {
    valid_matrix_error_input[0](0, 0) = 2.5f;
    valid_matrix_error_input[0](0, 1) = 4.5f;
    valid_matrix_error_input[0](0, 2) = 4.5f;
    valid_matrix_error_input[0](0, 3) = 4.5f;
    valid_matrix_error_input[0](0, 4) = 4.5f;
    valid_matrix_error_input[0](0, 5) = 2.f;
    valid_matrix_error_input[0](1, 0) = 4.25f;
    valid_matrix_error_input[0](1, 1) = 8.5f;
    valid_matrix_error_input[0](1, 2) = 8.5f;
    valid_matrix_error_input[0](1, 3) = 8.5f;
    valid_matrix_error_input[0](1, 4) = 8.5f;
    valid_matrix_error_input[0](1, 5) = 4.25f;
    valid_matrix_error_input[0](2, 0) = 4.25f;
    valid_matrix_error_input[0](2, 1) = 8.5f;
    valid_matrix_error_input[0](2, 2) = 8.5f;
    valid_matrix_error_input[0](2, 3) = 8.5f;
    valid_matrix_error_input[0](2, 4) = 8.5f;
    valid_matrix_error_input[0](2, 5) = 4.25f;
    valid_matrix_error_input[0](3, 0) = 4.25f;
    valid_matrix_error_input[0](3, 1) = 8.5f;
    valid_matrix_error_input[0](3, 2) = 8.5f;
    valid_matrix_error_input[0](3, 3) = 8.5f;
    valid_matrix_error_input[0](3, 4) = 8.5f;
    valid_matrix_error_input[0](3, 5) = 4.25f;
    valid_matrix_error_input[0](4, 0) = 4.25f;
    valid_matrix_error_input[0](4, 1) = 8.5f;
    valid_matrix_error_input[0](4, 2) = 8.5f;
    valid_matrix_error_input[0](4, 3) = 8.5f;
    valid_matrix_error_input[0](4, 4) = 8.5f;
    valid_matrix_error_input[0](4, 5) = 4.25f;
    valid_matrix_error_input[0](5, 0) = 1.75f;
    valid_matrix_error_input[0](5, 1) = 4.f;
    valid_matrix_error_input[0](5, 2) = 4.f;
    valid_matrix_error_input[0](5, 3) = 4.f;
    valid_matrix_error_input[0](5, 4) = 4.f;
    valid_matrix_error_input[0](5, 5) = 2.25f;

    valid_matrix_error_input[1](0, 0) = 4.f;
    valid_matrix_error_input[1](0, 1) = 7.f;
    valid_matrix_error_input[1](0, 2) = 7.f;
    valid_matrix_error_input[1](0, 3) = 7.f;
    valid_matrix_error_input[1](0, 4) = 7.f;
    valid_matrix_error_input[1](0, 5) = 3.f;
    valid_matrix_error_input[1](1, 0) = 6.5f;
    valid_matrix_error_input[1](1, 1) = 13.f;
    valid_matrix_error_input[1](1, 2) = 13.f;
    valid_matrix_error_input[1](1, 3) = 13.f;
    valid_matrix_error_input[1](1, 4) = 13.f;
    valid_matrix_error_input[1](1, 5) = 6.5f;
    valid_matrix_error_input[1](2, 0) = 6.5f;
    valid_matrix_error_input[1](2, 1) = 13.f;
    valid_matrix_error_input[1](2, 2) = 13.f;
    valid_matrix_error_input[1](2, 3) = 13.f;
    valid_matrix_error_input[1](2, 4) = 13.f;
    valid_matrix_error_input[1](2, 5) = 6.5f;
    valid_matrix_error_input[1](3, 0) = 6.5f;
    valid_matrix_error_input[1](3, 1) = 13.f;
    valid_matrix_error_input[1](3, 2) = 13.f;
    valid_matrix_error_input[1](3, 3) = 13.f;
    valid_matrix_error_input[1](3, 4) = 13.f;
    valid_matrix_error_input[1](3, 5) = 6.5f;
    valid_matrix_error_input[1](4, 0) = 6.5f;
    valid_matrix_error_input[1](4, 1) = 13.f;
    valid_matrix_error_input[1](4, 2) = 13.f;
    valid_matrix_error_input[1](4, 3) = 13.f;
    valid_matrix_error_input[1](4, 4) = 13.f;
    valid_matrix_error_input[1](4, 5) = 6.5f;
    valid_matrix_error_input[1](5, 0) = 2.5f;
    valid_matrix_error_input[1](5, 1) = 6.f;
    valid_matrix_error_input[1](5, 2) = 6.f;
    valid_matrix_error_input[1](5, 3) = 6.f;
    valid_matrix_error_input[1](5, 4) = 6.f;
    valid_matrix_error_input[1](5, 5) = 3.5f;

    valid_matrix_error_input[2](0, 0) = 19.f;
    valid_matrix_error_input[2](0, 1) = 38.f;
    valid_matrix_error_input[2](0, 2) = 38.f;
    valid_matrix_error_input[2](0, 3) = 38.f;
    valid_matrix_error_input[2](0, 4) = 38.f;
    valid_matrix_error_input[2](0, 5) = 19.f;
    valid_matrix_error_input[2](1, 0) = 38.f;
    valid_matrix_error_input[2](1, 1) = 76.f;
    valid_matrix_error_input[2](1, 2) = 76.f;
    valid_matrix_error_input[2](1, 3) = 76.f;
    valid_matrix_error_input[2](1, 4) = 76.f;
    valid_matrix_error_input[2](1, 5) = 38.f;
    valid_matrix_error_input[2](2, 0) = 38.f;
    valid_matrix_error_input[2](2, 1) = 76.f;
    valid_matrix_error_input[2](2, 2) = 76.f;
    valid_matrix_error_input[2](2, 3) = 76.f;
    valid_matrix_error_input[2](2, 4) = 76.f;
    valid_matrix_error_input[2](2, 5) = 38.f;
    valid_matrix_error_input[2](3, 0) = 38.f;
    valid_matrix_error_input[2](3, 1) = 76.f;
    valid_matrix_error_input[2](3, 2) = 76.f;
    valid_matrix_error_input[2](3, 3) = 76.f;
    valid_matrix_error_input[2](3, 4) = 76.f;
    valid_matrix_error_input[2](3, 5) = 38.f;
    valid_matrix_error_input[2](4, 0) = 38.f;
    valid_matrix_error_input[2](4, 1) = 76.f;
    valid_matrix_error_input[2](4, 2) = 76.f;
    valid_matrix_error_input[2](4, 3) = 76.f;
    valid_matrix_error_input[2](4, 4) = 76.f;
    valid_matrix_error_input[2](4, 5) = 38.f;
    valid_matrix_error_input[2](5, 0) = 19.f;
    valid_matrix_error_input[2](5, 1) = 38.f;
    valid_matrix_error_input[2](5, 2) = 38.f;
    valid_matrix_error_input[2](5, 3) = 38.f;
    valid_matrix_error_input[2](5, 4) = 38.f;
    valid_matrix_error_input[2](5, 5) = 19.f;

    valid_matrix_error_input[3](0, 0) = 22.f;
    valid_matrix_error_input[3](0, 1) = 44.f;
    valid_matrix_error_input[3](0, 2) = 44.f;
    valid_matrix_error_input[3](0, 3) = 44.f;
    valid_matrix_error_input[3](0, 4) = 44.f;
    valid_matrix_error_input[3](0, 5) = 22.f;
    valid_matrix_error_input[3](1, 0) = 44.f;
    valid_matrix_error_input[3](1, 1) = 88.f;
    valid_matrix_error_input[3](1, 2) = 88.f;
    valid_matrix_error_input[3](1, 3) = 88.f;
    valid_matrix_error_input[3](1, 4) = 88.f;
    valid_matrix_error_input[3](1, 5) = 44.f;
    valid_matrix_error_input[3](2, 0) = 44.f;
    valid_matrix_error_input[3](2, 1) = 88.f;
    valid_matrix_error_input[3](2, 2) = 88.f;
    valid_matrix_error_input[3](2, 3) = 88.f;
    valid_matrix_error_input[3](2, 4) = 88.f;
    valid_matrix_error_input[3](2, 5) = 44.f;
    valid_matrix_error_input[3](3, 0) = 44.f;
    valid_matrix_error_input[3](3, 1) = 88.f;
    valid_matrix_error_input[3](3, 2) = 88.f;
    valid_matrix_error_input[3](3, 3) = 88.f;
    valid_matrix_error_input[3](3, 4) = 88.f;
    valid_matrix_error_input[3](3, 5) = 44.f;
    valid_matrix_error_input[3](4, 0) = 44.f;
    valid_matrix_error_input[3](4, 1) = 88.f;
    valid_matrix_error_input[3](4, 2) = 88.f;
    valid_matrix_error_input[3](4, 3) = 88.f;
    valid_matrix_error_input[3](4, 4) = 88.f;
    valid_matrix_error_input[3](4, 5) = 44.f;
    valid_matrix_error_input[3](5, 0) = 22.f;
    valid_matrix_error_input[3](5, 1) = 44.f;
    valid_matrix_error_input[3](5, 2) = 44.f;
    valid_matrix_error_input[3](5, 3) = 44.f;
    valid_matrix_error_input[3](5, 4) = 44.f;
    valid_matrix_error_input[3](5, 5) = 22.f;
  }

  for (size_t ii = 1; ii < 4; ii++) {
    for (size_t i = 0; i < 6; i++) {
      for (size_t j = 0; j < 6; j++) {
        EXPECT_NEAR(valid_matrix_error_input[ii](i, j), errors_input[ii].toFloatMatrix(true)(i, j),
                    0.1f);
      }
    }
  }


  std::array<FloatMatrix, 4> valid_matrix_error_filter;
  for (auto &matrix : valid_matrix_error_filter) matrix = FloatMatrix(2, 2);
  {
    valid_matrix_error_filter[0](0, 0) = 171.f;
    valid_matrix_error_filter[0](0, 1) = 64.5f;
    valid_matrix_error_filter[0](1, 0) = 78.f;
    valid_matrix_error_filter[0](1, 1) = 69.f;

    valid_matrix_error_filter[1](0, 0) = 366.f;
    valid_matrix_error_filter[1](0, 1) = 150.5f;
    valid_matrix_error_filter[1](1, 0) = 182.f;
    valid_matrix_error_filter[1](1, 1) = 161.f;

    valid_matrix_error_filter[2](0, 0) = 561.f;
    valid_matrix_error_filter[2](0, 1) = 236.5f;
    valid_matrix_error_filter[2](1, 0) = 286.f;
    valid_matrix_error_filter[2](1, 1) = 253.f;

    valid_matrix_error_filter[3](0, 0) = 756.f;
    valid_matrix_error_filter[3](0, 1) = 322.5f;
    valid_matrix_error_filter[3](1, 0) = 390.f;
    valid_matrix_error_filter[3](1, 1) = 345.f;
  }

  for (size_t ii = 1; ii < 4; ii++) {
    for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 2; j++) {
        EXPECT_NEAR(valid_matrix_error_filter[ii](i, j),
                    storage.error_filter[ii].toFloatMatrix(true)(i, j), 0.1f);
      }
    }
  }
}

TEST(CNNLayerTest, canCreateAvgPoolingLayer) {
  CNNAvgPoolingLayer layer({4, 4}, {3, 3});

  ASSERT_EQ(false, layer.hasWeight());
  ASSERT_EQ(4, layer.getOutputSize().first);
  ASSERT_EQ(4, layer.getOutputSize().second);
}

TEST(CNNLayerTest, canCopyAvgPoolingLayer) {
  CNNAvgPoolingLayer layer({4, 4}, {3, 3});
  auto copy = layer.copy();
  ASSERT_EQ(copy->hasWeight(), layer.hasWeight());
  ASSERT_EQ(4, layer.getOutputSize().first);
  ASSERT_EQ(4, layer.getOutputSize().second);
}

TEST(CNNLayerTest, throwInvalidPoolingLayerWeight) {
  CNNAvgPoolingLayer layer({4, 4}, {3, 3});

  clFTensor tensor(3, 3, 1);
  ASSERT_ANY_THROW(layer.getWeight());
  ASSERT_ANY_THROW(layer.setWeight(tensor));
}

TEST(CNNLayerTest, canComputeAvgPoolingLayer) {
  auto& queue = utils::cl_wrapper.getDefaultQueue();
  CNNAvgPoolingLayer layer({4, 4}, {3, 3});

  math::FloatMatrix input1(6, 6);
  {
    input1(0, 0) = 1.f;
    input1(0, 1) = 2.f;
    input1(0, 2) = 1.f;
    input1(0, 3) = 1.f;
    input1(0, 4) = 4.f;
    input1(0, 5) = 1.f;
    input1(1, 0) = 2.f;
    input1(1, 1) = 1.f;
    input1(1, 2) = 1.f;
    input1(1, 3) = 2.f;
    input1(1, 4) = 2.f;
    input1(1, 5) = 1.f;
    input1(2, 0) = 4.f;
    input1(2, 1) = 3.f;
    input1(2, 2) = 2.f;
    input1(2, 3) = 1.f;
    input1(2, 4) = 2.f;
    input1(2, 5) = 1.f;
    input1(3, 0) = 1.f;
    input1(3, 1) = 5.f;
    input1(3, 2) = 1.f;
    input1(3, 3) = 1.f;
    input1(3, 4) = 2.f;
    input1(3, 5) = 1.f;
    input1(4, 0) = 2.f;
    input1(4, 1) = 1.f;
    input1(4, 2) = 1.f;
    input1(4, 3) = 4.f;
    input1(4, 4) = 1.f;
    input1(4, 5) = 1.f;
    input1(5, 0) = 2.f;
    input1(5, 1) = 1.f;
    input1(5, 2) = 4.f;
    input1(5, 3) = 2.f;
    input1(5, 4) = 4.f;
    input1(5, 5) = 1.f;
  }
  math::FloatMatrix input2(input1);
  input2(0, 0) = 100.f;

  clFTensor input_tensor(6, 6, 2);
  input_tensor[0] = input1;
  input_tensor[1] = input2;

  clFTensor output_tensor = layer.compute(queue, input_tensor);

  queue.finish();

  std::array<FloatMatrix, 2> valid_matrix;
  for (auto &matrix : valid_matrix) matrix = FloatMatrix(4, 4);

  {
    valid_matrix[0](0, 0) = 1.88f;
    valid_matrix[0](0, 1) = 1.55f;
    valid_matrix[0](0, 2) = 1.77f;
    valid_matrix[0](0, 3) = 1.66f;
    valid_matrix[0](1, 0) = 2.22f;
    valid_matrix[0](1, 1) = 1.88f;
    valid_matrix[0](1, 2) = 1.55f;
    valid_matrix[0](1, 3) = 1.44f;
    valid_matrix[0](2, 0) = 2.22f;
    valid_matrix[0](2, 1) = 2.11f;
    valid_matrix[0](2, 2) = 1.66f;
    valid_matrix[0](2, 3) = 1.55f;
    valid_matrix[0](3, 0) = 2.f;
    valid_matrix[0](3, 1) = 2.22f;
    valid_matrix[0](3, 2) = 2.22f;
    valid_matrix[0](3, 3) = 1.88f;

    valid_matrix[1](0, 0) = 12.88f;
    valid_matrix[1](0, 1) = 1.55f;
    valid_matrix[1](0, 2) = 1.77f;
    valid_matrix[1](0, 3) = 1.66f;
    valid_matrix[1](1, 0) = 2.22f;
    valid_matrix[1](1, 1) = 1.88f;
    valid_matrix[1](1, 2) = 1.55f;
    valid_matrix[1](1, 3) = 1.44f;
    valid_matrix[1](2, 0) = 2.22f;
    valid_matrix[1](2, 1) = 2.11f;
    valid_matrix[1](2, 2) = 1.66f;
    valid_matrix[1](2, 3) = 1.55f;
    valid_matrix[1](3, 0) = 2.f;
    valid_matrix[1](3, 1) = 2.22f;
    valid_matrix[1](3, 2) = 2.22f;
    valid_matrix[1](3, 3) = 1.88f;
  }
  for (size_t ii = 0; ii < 2; ii++) {
    for (size_t i = 0; i < 4; i++) {
      for (size_t j = 0; j < 4; j++) {
        EXPECT_NEAR(valid_matrix[ii](i, j), output_tensor[ii].toFloatMatrix(true)(i, j), 0.1f);
      }
    }
  }
}

TEST(CNNLayerTest, canComputeBPAvgPoolingLayer) {
  auto &queue = utils::cl_wrapper.getDefaultQueue();
  CNNAvgPoolingLayer layer({4, 4}, {3, 3});
  CNNStorageBPAvgPooling storage({6, 6});

  math::FloatMatrix input1(6, 6);
  {
    input1(0, 0) = 1.f;
    input1(0, 1) = 2.f;
    input1(0, 2) = 1.f;
    input1(0, 3) = 1.f;
    input1(0, 4) = 4.f;
    input1(0, 5) = 1.f;
    input1(1, 0) = 2.f;
    input1(1, 1) = 1.f;
    input1(1, 2) = 1.f;
    input1(1, 3) = 2.f;
    input1(1, 4) = 2.f;
    input1(1, 5) = 1.f;
    input1(2, 0) = 4.f;
    input1(2, 1) = 3.f;
    input1(2, 2) = 2.f;
    input1(2, 3) = 1.f;
    input1(2, 4) = 2.f;
    input1(2, 5) = 1.f;
    input1(3, 0) = 1.f;
    input1(3, 1) = 5.f;
    input1(3, 2) = 1.f;
    input1(3, 3) = 1.f;
    input1(3, 4) = 2.f;
    input1(3, 5) = 1.f;
    input1(4, 0) = 2.f;
    input1(4, 1) = 1.f;
    input1(4, 2) = 1.f;
    input1(4, 3) = 4.f;
    input1(4, 4) = 1.f;
    input1(4, 5) = 1.f;
    input1(5, 0) = 2.f;
    input1(5, 1) = 1.f;
    input1(5, 2) = 4.f;
    input1(5, 3) = 2.f;
    input1(5, 4) = 4.f;
    input1(5, 5) = 1.f;
  }
  math::FloatMatrix input2(input1);
  input2(0, 0) = 100.f;

  clFTensor input_tensor(6, 6, 2);
  input_tensor[0] = input1;
  input_tensor[1] = input2;

  clFTensor output_tensor = layer.computeForward(queue, input_tensor, storage);

  queue.finish();

  std::array<FloatMatrix, 2> valid_matrix_output;
  for (auto &matrix : valid_matrix_output) matrix = FloatMatrix(4, 4);
  {
    valid_matrix_output[0](0, 0) = 1.88f;
    valid_matrix_output[0](0, 1) = 1.55f;
    valid_matrix_output[0](0, 2) = 1.77f;
    valid_matrix_output[0](0, 3) = 1.66f;
    valid_matrix_output[0](1, 0) = 2.22f;
    valid_matrix_output[0](1, 1) = 1.88f;
    valid_matrix_output[0](1, 2) = 1.55f;
    valid_matrix_output[0](1, 3) = 1.44f;
    valid_matrix_output[0](2, 0) = 2.22f;
    valid_matrix_output[0](2, 1) = 2.11f;
    valid_matrix_output[0](2, 2) = 1.66f;
    valid_matrix_output[0](2, 3) = 1.55f;
    valid_matrix_output[0](3, 0) = 2.f;
    valid_matrix_output[0](3, 1) = 2.22f;
    valid_matrix_output[0](3, 2) = 2.22f;
    valid_matrix_output[0](3, 3) = 1.88f;

    valid_matrix_output[1](0, 0) = 12.88f;
    valid_matrix_output[1](0, 1) = 1.55f;
    valid_matrix_output[1](0, 2) = 1.77f;
    valid_matrix_output[1](0, 3) = 1.66f;
    valid_matrix_output[1](1, 0) = 2.22f;
    valid_matrix_output[1](1, 1) = 1.88f;
    valid_matrix_output[1](1, 2) = 1.55f;
    valid_matrix_output[1](1, 3) = 1.44f;
    valid_matrix_output[1](2, 0) = 2.22f;
    valid_matrix_output[1](2, 1) = 2.11f;
    valid_matrix_output[1](2, 2) = 1.66f;
    valid_matrix_output[1](2, 3) = 1.55f;
    valid_matrix_output[1](3, 0) = 2.f;
    valid_matrix_output[1](3, 1) = 2.22f;
    valid_matrix_output[1](3, 2) = 2.22f;
    valid_matrix_output[1](3, 3) = 1.88f;
  }

  for (size_t ii = 0; ii < 2; ii++) {
    for (size_t i = 0; i < 4; i++) {
      for (size_t j = 0; j < 4; j++) {
        EXPECT_NEAR(valid_matrix_output[ii](i, j), output_tensor[ii].toFloatMatrix(true)(i, j),
                    0.1f);
      }
    }
  }

  clFTensor errors_tensor(4, 4, 4);
  errors_tensor[0].fill(1.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[1].fill(2.f, utils::cl_wrapper.getDefaultQueue(), true);

  clFTensor errors_input = layer.computeBackward(queue, errors_tensor, storage);

  queue.finish();

  std::array<FloatMatrix, 2> valid_matrix_error_input;
  for (auto &matrix : valid_matrix_error_input) matrix = FloatMatrix(6, 6);
  {
    valid_matrix_error_input[0](0, 0) = 0.111111f;
    valid_matrix_error_input[0](0, 1) = 0.222222f;
    valid_matrix_error_input[0](0, 2) = 0.333333f;
    valid_matrix_error_input[0](0, 3) = 0.333333f;
    valid_matrix_error_input[0](0, 4) = 0.222222f;
    valid_matrix_error_input[0](0, 5) = 0.111111f;
    valid_matrix_error_input[0](1, 0) = 0.222222f;
    valid_matrix_error_input[0](1, 1) = 0.444444f;
    valid_matrix_error_input[0](1, 2) = 0.666667f;
    valid_matrix_error_input[0](1, 3) = 0.666667f;
    valid_matrix_error_input[0](1, 4) = 0.444444f;
    valid_matrix_error_input[0](1, 5) = 0.222222f;
    valid_matrix_error_input[0](2, 0) = 0.333333f;
    valid_matrix_error_input[0](2, 1) = 0.666667f;
    valid_matrix_error_input[0](2, 2) = 1.f;
    valid_matrix_error_input[0](2, 3) = 1.f;
    valid_matrix_error_input[0](2, 4) = 0.666667f;
    valid_matrix_error_input[0](2, 5) = 0.333333f;
    valid_matrix_error_input[0](3, 0) = 0.333333f;
    valid_matrix_error_input[0](3, 1) = 0.666667f;
    valid_matrix_error_input[0](3, 2) = 1.f;
    valid_matrix_error_input[0](3, 3) = 1.f;
    valid_matrix_error_input[0](3, 4) = 0.666667f;
    valid_matrix_error_input[0](3, 5) = 0.333333f;
    valid_matrix_error_input[0](4, 0) = 0.222222f;
    valid_matrix_error_input[0](4, 1) = 0.444444f;
    valid_matrix_error_input[0](4, 2) = 0.666667f;
    valid_matrix_error_input[0](4, 3) = 0.666667f;
    valid_matrix_error_input[0](4, 4) = 0.444444f;
    valid_matrix_error_input[0](4, 5) = 0.222222f;
    valid_matrix_error_input[0](5, 0) = 0.111111f;
    valid_matrix_error_input[0](5, 1) = 0.222222f;
    valid_matrix_error_input[0](5, 2) = 0.333333f;
    valid_matrix_error_input[0](5, 3) = 0.333333f;
    valid_matrix_error_input[0](5, 4) = 0.222222f;
    valid_matrix_error_input[0](5, 5) = 0.111111f;

    valid_matrix_error_input[1](0, 0) = 0.222222f;
    valid_matrix_error_input[1](0, 1) = 0.444444f;
    valid_matrix_error_input[1](0, 2) = 0.666667f;
    valid_matrix_error_input[1](0, 3) = 0.666667f;
    valid_matrix_error_input[1](0, 4) = 0.444444f;
    valid_matrix_error_input[1](0, 5) = 0.222222f;
    valid_matrix_error_input[1](1, 0) = 0.444444f;
    valid_matrix_error_input[1](1, 1) = 0.888889f;
    valid_matrix_error_input[1](1, 2) = 1.33333f;
    valid_matrix_error_input[1](1, 3) = 1.33333f;
    valid_matrix_error_input[1](1, 4) = 0.888889f;
    valid_matrix_error_input[1](1, 5) = 0.444444f;
    valid_matrix_error_input[1](2, 0) = 0.666667f;
    valid_matrix_error_input[1](2, 1) = 1.33333f;
    valid_matrix_error_input[1](2, 2) = 2.f;
    valid_matrix_error_input[1](2, 3) = 2.f;
    valid_matrix_error_input[1](2, 4) = 1.33333f;
    valid_matrix_error_input[1](2, 5) = 0.666667f;
    valid_matrix_error_input[1](3, 0) = 0.666667f;
    valid_matrix_error_input[1](3, 1) = 1.33333f;
    valid_matrix_error_input[1](3, 2) = 2.f;
    valid_matrix_error_input[1](3, 3) = 2.f;
    valid_matrix_error_input[1](3, 4) = 1.33333f;
    valid_matrix_error_input[1](3, 5) = 0.666667f;
    valid_matrix_error_input[1](4, 0) = 0.444444f;
    valid_matrix_error_input[1](4, 1) = 0.888889f;
    valid_matrix_error_input[1](4, 2) = 1.33333f;
    valid_matrix_error_input[1](4, 3) = 1.33333f;
    valid_matrix_error_input[1](4, 4) = 0.888889f;
    valid_matrix_error_input[1](4, 5) = 0.444444f;
    valid_matrix_error_input[1](5, 0) = 0.222222f;
    valid_matrix_error_input[1](5, 1) = 0.444444f;
    valid_matrix_error_input[1](5, 2) = 0.666667f;
    valid_matrix_error_input[1](5, 3) = 0.666667f;
    valid_matrix_error_input[1](5, 4) = 0.444444f;
    valid_matrix_error_input[1](5, 5) = 0.222222f;
  }

  for (size_t ii = 0; ii < 2; ii++) {
    for (size_t i = 0; i < 6; i++) {
      for (size_t j = 0; j < 6; j++) {
        EXPECT_NEAR(valid_matrix_error_input[ii](i, j), errors_input[ii].toFloatMatrix(true)(i, j),
                    0.1f);
      }
    }
  }
}

TEST(CNNBackpropStorage, canCreateBPStorage) {
  CNNStorageBPConvolution convolution_storage;
  ASSERT_EQ(true, convolution_storage.hasGradient());

  CNNStorageBPAvgPooling pooling_storage({10, 10});
  ASSERT_EQ(10, pooling_storage.input_size.first);
  ASSERT_EQ(10, pooling_storage.input_size.second);
  ASSERT_EQ(false, pooling_storage.hasGradient());
}

TEST(CNNBackpropStorage, throwInvalidBPStorage) {
  CNNStorageBPAvgPooling pooling_storage({10, 10});
  ASSERT_ANY_THROW(pooling_storage.getGradient());
}

TEST(CNNTest, canCreateCnn) {
  std::string str_topology("6 6 relu convolution 2 2 2 convolution 2 2 2 pooling avg 2 2");
  auto topology = stringToTopology(str_topology);

  CNN cnn;
  cnn.setTopology(topology);

  ASSERT_EQ(3, cnn.getLayers().size());
  ASSERT_EQ(3, cnn.copyLayers().size());
}

TEST(CNNTest, canPredictCnn) {
  auto &queue = utils::cl_wrapper.getDefaultQueue();
  std::string str_topology("6 6 relu convolution 2 2 2 convolution 2 2 2 pooling avg 2 2");
  auto topology = stringToTopology(str_topology);

  CNN cnn;
  cnn.setTopology(topology);

  FloatMatrix f1(2, 2);
  {
    f1(0, 0) = 2.f;
    f1(0, 1) = 1.f;
    f1(1, 0) = 0.5f;
    f1(1, 1) = 1.5f;
  }
  FloatMatrix f2(2, 2);
  {
    f2(0, 0) = 2.f;
    f2(0, 1) = 1.f;
    f2(1, 0) = 0.f;
    f2(1, 1) = 1.f;
  }
  math::FloatMatrix input1(6, 6);
  {
    input1(0, 0) = 1.f;
    input1(0, 1) = 2.f;
    input1(0, 2) = 1.f;
    input1(0, 3) = 1.f;
    input1(0, 4) = 4.f;
    input1(0, 5) = 1.f;
    input1(1, 0) = 2.f;
    input1(1, 1) = 1.f;
    input1(1, 2) = 1.f;
    input1(1, 3) = 2.f;
    input1(1, 4) = 2.f;
    input1(1, 5) = 1.f;
    input1(2, 0) = 4.f;
    input1(2, 1) = 3.f;
    input1(2, 2) = 2.f;
    input1(2, 3) = 1.f;
    input1(2, 4) = 2.f;
    input1(2, 5) = 1.f;
    input1(3, 0) = 1.f;
    input1(3, 1) = 5.f;
    input1(3, 2) = 1.f;
    input1(3, 3) = 1.f;
    input1(3, 4) = 2.f;
    input1(3, 5) = 1.f;
    input1(4, 0) = 2.f;
    input1(4, 1) = 1.f;
    input1(4, 2) = 1.f;
    input1(4, 3) = 4.f;
    input1(4, 4) = 1.f;
    input1(4, 5) = 1.f;
    input1(5, 0) = 2.f;
    input1(5, 1) = 1.f;
    input1(5, 2) = 4.f;
    input1(5, 3) = 2.f;
    input1(5, 4) = 4.f;
    input1(5, 5) = 1.f;
  }
  math::FloatMatrix input2(input1);
  input2(0, 0) = 100.f;

  const auto &layer1 = dynamic_cast<const CNNConvolutionLayer *>(cnn.getLayers()[0].get());
  auto &filter1 = layer1->getFilter();
  filter1[0] = f1;
  filter1[1] = f2;

  const auto &layer2 = dynamic_cast<const CNNConvolutionLayer *>(cnn.getLayers()[1].get());
  auto &filter2 = layer2->getFilter();
  filter2[0] = f1;
  filter2[1] = f2;
  filter2[2] = f1;
  filter2[3] = f2;

  clFTensor input_tensor(6, 6, 2);
  input_tensor[0] = input1;
  input_tensor[1] = input2;

  clFTensor output_tensor = cnn.predict(queue, input_tensor);

  queue.finish();

  std::array<FloatMatrix, 2> valid_matrix;
  for (auto &matrix : valid_matrix) matrix = FloatMatrix(36, 1);

  {
    valid_matrix[0](0, 0) = 41.75f;
    valid_matrix[0](1, 0) = 37.4375f;
    valid_matrix[0](2, 0) = 41.6875f;
    valid_matrix[0](3, 0) = 55.5f;
    valid_matrix[0](4, 0) = 41.125f;
    valid_matrix[0](5, 0) = 38.25f;
    valid_matrix[0](6, 0) = 58.1875f;
    valid_matrix[0](7, 0) = 50.5f;
    valid_matrix[0](8, 0) = 42.5f;
    valid_matrix[0](9, 0) = 31.375f;
    valid_matrix[0](10, 0) = 29.5f;
    valid_matrix[0](11, 0) = 33.875f;
    valid_matrix[0](12, 0) = 43.625f;
    valid_matrix[0](13, 0) = 32.f;
    valid_matrix[0](14, 0) = 30.625f;
    valid_matrix[0](15, 0) = 47.875f;
    valid_matrix[0](16, 0) = 39.625f;
    valid_matrix[0](17, 0) = 32.375f;
    valid_matrix[0](18, 0) = 31.375f;
    valid_matrix[0](19, 0) = 29.5f;
    valid_matrix[0](20, 0) = 33.875f;
    valid_matrix[0](21, 0) = 43.625f;
    valid_matrix[0](22, 0) = 32.f;
    valid_matrix[0](23, 0) = 30.625f;
    valid_matrix[0](24, 0) = 47.875f;
    valid_matrix[0](25, 0) = 39.625f;
    valid_matrix[0](26, 0) = 32.375f;
    valid_matrix[0](27, 0) = 23.5f;
    valid_matrix[0](28, 0) = 23.25f;
    valid_matrix[0](29, 0) = 27.5f;
    valid_matrix[0](30, 0) = 33.75f;
    valid_matrix[0](31, 0) = 24.75f;
    valid_matrix[0](32, 0) = 24.75f;
    valid_matrix[0](33, 0) = 39.5f;
    valid_matrix[0](34, 0) = 31.25f;
    valid_matrix[0](35, 0) = 24.75f;

    valid_matrix[1](0, 0) = 140.75f;
    valid_matrix[1](1, 0) = 37.4375f;
    valid_matrix[1](2, 0) = 41.6875f;
    valid_matrix[1](3, 0) = 55.5f;
    valid_matrix[1](4, 0) = 41.125f;
    valid_matrix[1](5, 0) = 38.25f;
    valid_matrix[1](6, 0) = 58.1875f;
    valid_matrix[1](7, 0) = 50.5f;
    valid_matrix[1](8, 0) = 42.5f;
    valid_matrix[1](9, 0) = 130.375f;
    valid_matrix[1](10, 0) = 29.5f;
    valid_matrix[1](11, 0) = 33.875f;
    valid_matrix[1](12, 0) = 43.625f;
    valid_matrix[1](13, 0) = 32.f;
    valid_matrix[1](14, 0) = 30.625f;
    valid_matrix[1](15, 0) = 47.875f;
    valid_matrix[1](16, 0) = 39.625f;
    valid_matrix[1](17, 0) = 32.375f;
    valid_matrix[1](18, 0) = 130.375f;
    valid_matrix[1](19, 0) = 29.5f;
    valid_matrix[1](20, 0) = 33.875f;
    valid_matrix[1](21, 0) = 43.625f;
    valid_matrix[1](22, 0) = 32.f;
    valid_matrix[1](23, 0) = 30.625f;
    valid_matrix[1](24, 0) = 47.875f;
    valid_matrix[1](25, 0) = 39.625f;
    valid_matrix[1](26, 0) = 32.375f;
    valid_matrix[1](27, 0) = 122.5f;
    valid_matrix[1](28, 0) = 23.25f;
    valid_matrix[1](29, 0) = 27.5f;
    valid_matrix[1](30, 0) = 33.75f;
    valid_matrix[1](31, 0) = 24.75f;
    valid_matrix[1](32, 0) = 24.75f;
    valid_matrix[1](33, 0) = 39.5f;
    valid_matrix[1](34, 0) = 31.25f;
    valid_matrix[1](35, 0) = 24.75f;
  }
  for (size_t ii = 0; ii < 2; ii++) {
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 4; j++) {
        EXPECT_NEAR(valid_matrix[ii](i, j), output_tensor[ii].toFloatMatrix(true)(i, j), 0.1f);
      }
    }
  }
}


int main(int argc, char **argv) {
  utils::clWrapper::initOpenCL(*utils::clWrapper::makeDefault("../kernels"));
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}