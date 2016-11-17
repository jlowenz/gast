#include <gast/pygast.hpp>
#include <gast/gast.hpp>

struct symmetry_impl_t
{
  gast::symmetry_state_t state;
};

Symmetry::Symmetry()
  : state_(new symmetry_impl_t)
{
}

Symmetry::~Symmetry()
{
  delete state_;
}

void Symmetry::transform(unsigned char* img, int irows, int icols,
			 float* sym, int srows, int scols, 
			 float* dir, int drows, int dcols,
			 int sigma)
{
  cv::Mat _img(irows, icols, CV_8U, img);
  cv::Mat _sym(srows, scols, CV_32F, sym);
  cv::Mat _dir(drows, dcols, CV_32F, dir);
  state_->state.sigma = sigma;
  symmetry_transform(_img, _sym, _dir, state_->state);
}

