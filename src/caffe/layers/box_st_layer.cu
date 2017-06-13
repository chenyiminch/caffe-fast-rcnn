#include <cfloat>

#include "caffe/vision_layers.hpp"

namespace caffe
{

template <typename Dtype>
void BoxSpatialTransformerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
	// const Dtype* bottom_data = bottom[0]->gpu_data();
	// const Dtype* theta_data = bottom[1]->gpu_data();

	// Dtype* top_data = top[0]->mutable_gpu_data();

	NOT_IMPLEMENTED;

}

template <typename Dtype>
void BoxSpatialTransformerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(BoxSpatialTransformerLayer);
}