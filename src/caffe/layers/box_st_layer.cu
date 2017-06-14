#include <cfloat>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe
{

template <typename Dtype>
__global__ void BoxTransformForward(const int nthreads, const Dtype* bottom_theta_data, const Dtype* bottom_coordinate_data,
	Dtype* top_data)
{
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		bottom_theta_data += index * 4;
		bottom_coordinate_data += index * 4;
		top_data[index * 4] = bottom_theta_data[0] * bottom_coordinate_data[0] + bottom_theta_data[2];
		top_data[index * 4 + 1] = bottom_theta_data[1] * bottom_coordinate_data[1] + bottom_theta_data[3];

		top_data[index * 4 + 2] = bottom_theta_data[0] * bottom_coordinate_data[2] + bottom_theta_data[2];
		top_data[index * 4 + 3] = bottom_theta_data[1] * bottom_coordinate_data[3] + bottom_theta_data[3];
	}
}

template <typename Dtype>
void BoxSpatialTransformerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
	const Dtype* bottom_coordinate_data = bottom[0]->gpu_data();
	const Dtype* bottom_theta_data = bottom[1]->gpu_data();

	Dtype* top_data = top[0]->mutable_gpu_data();

	caffe_gpu_set(top[0]->count(), (Dtype)0, top_data);

	int batch_size = top[0]->shape(0);

	BoxTransformForward<Dtype><<<CAFFE_GET_BLOCKS(batch_size), CAFFE_CUDA_NUM_THREADS>>>(
		batch_size, bottom_theta_data, bottom_coordinate_data, top_data);
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void BoxTransformBackward(const int nthreads, const Dtype* top_diff, const Dtype* bottom_coordinate_data,
	Dtype* bottom_theta_diff)
{
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		bottom_coordinate_data += index * 4;
		top_diff += index * 4;
		bottom_theta_diff[index * 4] = top_diff[0] * bottom_coordinate_data[0] + top_diff[2] * bottom_coordinate_data[2];
		bottom_theta_diff[index * 4 + 1] = top_diff[1] * bottom_coordinate_data[1] + top_diff[3] * bottom_coordinate_data[3];
		bottom_theta_diff[index * 4 + 2] = top_diff[0] + top_diff[2];
		bottom_theta_diff[index * 4 + 3] = top_diff[1] + top_diff[3];
	}
}

template <typename Dtype>
void BoxSpatialTransformerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	Dtype* bottom_theta_diff = bottom[1]->mutable_gpu_diff();
	const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* bottom_coordinate_data = bottom[0]->gpu_data();

	caffe_gpu_set(bottom[1]->count(), (Dtype)0, bottom_theta_diff);

	int batch_size = top[0]->shape(0);

	BoxTransformBackward<Dtype><<<CAFFE_GET_BLOCKS(batch_size), CAFFE_CUDA_NUM_THREADS>>>(
		batch_size, top_diff, bottom_coordinate_data, bottom_theta_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(BoxSpatialTransformerLayer);
}