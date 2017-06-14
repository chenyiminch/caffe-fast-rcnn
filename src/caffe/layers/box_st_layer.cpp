#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;

namespace caffe
{

template <typename Dtype>
void BoxSpatialTransformerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void BoxSpatialTransformerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
	vector<int> output_shape(4);
	output_shape[0] = 1;
	output_shape[1] = 4;
	output_shape[2] = 1;
	output_shape[3] = 1;

	top[0]->Reshape(output_shape);

	vector<int> theta_shape(2);
	theta_shape[0] = 2;
	theta_shape[1] = 3;

	theta_blob.Reshape(theta_shape);

	vector<int> coordinate_shape(2);
	coordinate_shape[0] = 3;
	coordinate_shape[1] = 2;

	coordinate_blob.Reshape(coordinate_shape);
}

template <typename Dtype>
void BoxSpatialTransformerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
	// NOT_IMPLEMENTED;
	int batch_size = bottom[1]->shape(0);
	int num_thetas = bottom[1]->shape(1);
	int theta_height = bottom[1]->shape(2);
	int theta_width = bottom[1]->shape(3);
	CHECK(num_thetas * theta_height * theta_width == 4) 
	<< "number of parameters for the spatial transformation should be equal to 4" << endl;

	const Dtype* input_coordinates = bottom[0]->cpu_data();
	const Dtype* theta_data = bottom[1]->cpu_data();

	Dtype* output_coordinates = top[0]->mutable_cpu_data();

	caffe_set(top[0]->count(), (Dtype)0, output_coordinates);

	Dtype* theta_blob_data = theta_blob.mutable_cpu_data();
	caffe_set(theta_blob.count(), (Dtype)0, theta_blob_data);

	Dtype* coordinate_blob_data = coordinate_blob.mutable_cpu_data();
	caffe_set(coordinate_blob.count(), (Dtype)1, coordinate_blob_data);

	for (int i = 0; i < batch_size; ++i)
	{
		Dtype* coordinates = output_coordinates + i * 4;
		theta_blob_data[0] = theta_data[i * 4];
		theta_blob_data[3] = theta_data[i * 4 + 1];
		theta_blob_data[4] = theta_data[i * 4 + 2];
		theta_blob_data[5] = theta_data[i * 4 + 3];

		for (int j = 0; j < 2; ++j)
		{
			for (int k = 0; k < 2; ++k)
			{
				coordinate_blob_data[j * 3 + k] = input_coordinates[j * 2 + k];
			}
		}

		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 2, 2, 3, (Dtype)1, 
			theta_blob_data, coordinate_blob_data, (Dtype)0, coordinates);
	}
}

template <typename Dtype>
void BoxSpatialTransformerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	// NOT_IMPLEMENTED;
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* theta_diff = bottom[1]->mutable_cpu_diff();
	caffe_set(bottom[1]->count(), (Dtype)0, theta_diff);

	int batch_size = bottom[0]->shape(0);
	const Dtype* bottom_data = bottom[0]->cpu_data();
	for (int i = 0; i < batch_size; ++i)
	{
		theta_diff[i * 4] = top_diff[i * 4] * bottom_data[i * 4] + top_diff[i * 4 + 2] * bottom_data[i * 4 + 2];
		theta_diff[i * 4 + 1] = top_diff[i * 4 + 1] * bottom_data[i * 4 + 1] + top_diff[i * 4 + 3] * bottom_data[i * 4 + 3];
		theta_diff[i * 4 + 2] = top_diff[i * 4] + top_diff[i * 4 + 2];
		theta_diff[i * 4 + 3] = top_diff[i * 4 + 1] + top_diff[i * 4 + 3];
	}
}

#ifdef CPU_ONLY
STUB_GPU(BoxSpatialTransformerLayer);
#endif

INSTANTIATE_CLASS(BoxSpatialTransformerLayer);
REGISTER_LAYER_CLASS(BoxSpatialTransformer);

}