#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using namespace std;

namespace caffe
{

typedef ::testing::Types<FloatGPU, DoubleGPU> TestDtypesGPU;

template <typename TypeParam>
	class BoxSpatialTransformerLayerTest : public MultiDeviceTest<TypeParam>
	{
		typedef typename TypeParam::Dtype Dtype;
	protected:
		BoxSpatialTransformerLayerTest()
			: blob_bottom_box_(new Blob<Dtype>(1, 4, 1, 1)),
			blob_bottom_theta_(new Blob<Dtype>(1, 4, 1, 1)),
			blob_top_(new Blob<Dtype>()) {
				// FillerParameter filler_param;
    //   			GaussianFiller<Dtype> filler(filler_param);
    //   			filler.Fill(this->blob_bottom_box_);
    //   			blob_bottom_vec_.push_back(blob_bottom_box_);
    //   			filler.Fill(this->blob_bottom_theta_);
      			for (int i = 0; i < blob_bottom_theta_->count(); ++i) {
     				blob_bottom_theta_->mutable_cpu_data()[i] = i;
   	 			}
   	 			for (int i = 0; i < blob_bottom_box_->count(); ++i) {
     				blob_bottom_box_->mutable_cpu_data()[i] = i+2;
   	 			}
				blob_bottom_vec_.push_back(blob_bottom_box_);
				blob_bottom_vec_.push_back(blob_bottom_theta_);
				blob_top_vec_.push_back(blob_top_);
			}
		virtual ~BoxSpatialTransformerLayerTest() {
			delete blob_bottom_box_;
			delete blob_bottom_theta_;
			delete blob_top_;
		}

		Blob<Dtype>* const blob_bottom_box_;
		Blob<Dtype>* const blob_bottom_theta_;
		Blob<Dtype>* const blob_top_;

		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(BoxSpatialTransformerLayerTest, TestDtypesAndDevices);
	// TYPED_TEST_CASE(BoxSpatialTransformerLayerTest, TestDtypesGPU);

	TYPED_TEST(BoxSpatialTransformerLayerTest, TestForward)
	{
		typedef typename TypeParam::Dtype Dtype;
    	LayerParameter layer_param;

    	BoxSpatialTransformerLayer<Dtype> layer(layer_param);
    	layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    	layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    	for (int i = 0; i < 4; ++i)
    	{
    		Dtype data = this->blob_top_->data_at(0, i, 0, 0);
    		LOG(INFO) << data;
    		if (i == 0)
    		{
    			const Dtype kErrorBound = 0.001;
    			EXPECT_NEAR(2, data, kErrorBound);
    		}
    		if (i == 1)
    		{
    			const Dtype kErrorBound = 0.001;
    			EXPECT_NEAR(6, data, kErrorBound);
    		}
    		if (i == 2)
    		{
    			const Dtype kErrorBound = 0.001;
    			EXPECT_NEAR(2, data, kErrorBound);
    		}
    		if (i == 3)
    		{
    			const Dtype kErrorBound = 0.001;
    			EXPECT_NEAR(8, data, kErrorBound);
    		}
    	}
	}

	TYPED_TEST(BoxSpatialTransformerLayerTest, TestGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;

    BoxSpatialTransformerLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-4);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 1);
  }
}