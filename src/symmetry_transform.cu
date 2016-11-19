#include <gast/gast.hpp>
#include <gast/pygast.hpp>
//#include <opencv2/gpu/gpu.hpp>
#include <gpu_buffers/gpu_buffers.hpp>
#include <opencv2/gpu/stream_accessor.hpp>
#include <math_functions.h>

using namespace gpu_buffers;

namespace gast {

  __global__ void cu_gradient(const cv::gpu::PtrStepSzf dx_,
			      const cv::gpu::PtrStepSzf dy_,
			      cv::gpu::PtrStepSzf mag, 
			      cv::gpu::PtrStepSzf dir)
  {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dx_.cols || y >= dx_.rows) return;
    float dx = dx_.ptr(y)[x];
    float dy = dy_.ptr(y)[x];
    mag.ptr(y)[x] = log(1.f + sqrtf(dx*dx + dy*dy));
    dir.ptr(y)[x] = atan2f(dy, dx);
  }

  void gradient(const cv::gpu::GpuMat& dx, const cv::gpu::GpuMat& dy,
		cv::gpu::GpuMat& mag, cv::gpu::GpuMat& dir, cv::gpu::Stream& s)
  {
    cv::gpu::createContinuous(dx.size(), dx.type(), mag);
    cv::gpu::createContinuous(dy.size(), dy.type(), dir);
    dim3 cthreads(32,8);
    dim3 cblocks(divUp(dx.cols,cthreads.x),
		 divUp(dy.rows,cthreads.y));
    cudaStream_t cs = cv::gpu::StreamAccessor::getStream(s);
    cu_gradient<<<cblocks, cthreads, 0, cs>>>(dx, dy, mag, dir);
  }

  __device__ inline int2 operator-(const int2& a, const int2& b)
  {
    return make_int2(a.x-b.x, a.y-b.y);
  }

  __device__ inline float norm(const int2& p)
  {
    return sqrtf(p.x*p.x + p.y*p.y);
  }

  __device__ inline bool operator==(const int2& a, const int2& b)
  {
    return a.x == b.x && a.y == b.y;
  }

  __device__ inline float dist(const int2& pti, const int2& ptj, float sigma)
  {
    const float sqrt2piinv = 0.39894228;
    float scale = 1.f / sigma * sqrt2piinv;
    float sigma2 = 2.f * sigma;
    float diff = norm(pti - ptj);
    return scale * exp(-diff/sigma2);
  }

  __device__ inline float phase(float alpha_ij, float theta_i, float theta_j)
  {
    return (1 - cos(theta_i + theta_j - 2*alpha_ij)) * (1 - cos(theta_i - theta_j));
  }

  __device__ inline float2 pt_gradient(const cv::gpu::PtrStepSzf& mag,
				       const cv::gpu::PtrStepSzf& dir, 
				       const int2& p)
  {
    return make_float2(mag(p.y,p.x), dir(p.y,p.x));
  }

  __device__ inline bool valid_pt(int2 p, int rows, int cols)
  {
    return (p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows);
  }

  //__device__ inline 

  /*
    To compute the symmetry value for pixel p, we need to look at all
    the surrounding pixels (Gamma in the paper) in radius 3*sigma
  */
  __global__ void cu_symmetry(cv::gpu::PtrStepSzf smag,
			      cv::gpu::PtrStepSzf sdir,
			      const cv::gpu::PtrStepSzf mag,
			      const cv::gpu::PtrStepSzf dir, 			    
			      int sigma)
  {
    // prepare region for given pixel x,y
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    //int x_radius = min(3 * sigma, min(x, mag.cols - x - 1));
    //int y_radius = min(3 * sigma, min(y, mag.rows - y - 1));
    int radius = (int)(2.5 * sigma);
    int ROWS = mag.rows;
    int COLS = mag.cols;
  
    // target point, we are computing the symmetry for THIS point
    int2 p = make_int2(x, y);
    if (p.x >= COLS || p.y >= ROWS) return;

    int2 min = make_int2(p.x - radius, p.y - radius);
    int2 max = make_int2(p.x + radius + 1, p.y + radius + 1);
    int2 pi, pj, d;
    float2 rtheta_i, rtheta_j;
    float M = 0;
    float RS = 0;
    float C_ij = 0;
    float psi_ij = 0;
    float maxC = 0;
    float maxTheta = 0;
    float alpha_ij = 0;
    for (int j = min.y; j <= p.y; j++) {
      for (int i = min.x; i < max.x; i++) {
	if (abs(i-p.x) < sigma &&
	    abs(j-p.y) < sigma) continue;
    	pi = make_int2(i,j);
    	if (pi == p) break; // we are done, since this computation is symmetric
    	d = pi - p;
    	pj = p - d;
    	if (valid_pt(pi, ROWS, COLS) && valid_pt(pj, ROWS, COLS)) {
    	  rtheta_i = pt_gradient(mag, dir, pi);
    	  rtheta_j = pt_gradient(mag, dir, pj);	
    	  alpha_ij = atan2f(pi.y - pj.y, pi.x - pj.x);
    	  C_ij = rtheta_i.x * rtheta_j.x * dist(pi, pj, (float)sigma) * 
    	    phase(alpha_ij, rtheta_i.y, rtheta_j.y);	  
    	  M += C_ij;
	  psi_ij = (rtheta_i.y + rtheta_j.y) * 0.5;
    	  if (C_ij > maxC) {
    	    maxC = C_ij;
    	    maxTheta = psi_ij;
    	  }
	  float sin_phi = sin(psi_ij - maxTheta);
	  RS += C_ij * (sin_phi * sin_phi);
    	}
      }
    }
    smag(p.y,p.x) = M;
    sdir(p.y,p.x) = maxTheta;
  }

  void symmetry(cv::gpu::GpuMat& smag, cv::gpu::GpuMat& sdir,
		const cv::gpu::GpuMat& mag, const cv::gpu::GpuMat& dir,
		int sigma, cv::gpu::Stream& s)
  {
    cv::gpu::createContinuous(mag.size(), CV_32F, smag);
    cv::gpu::createContinuous(mag.size(), CV_32F, sdir);
    dim3 cthreads(16,16);
    dim3 cblocks(divUp(mag.cols,cthreads.x),
		 divUp(mag.rows,cthreads.y));
    cudaStream_t cs = cv::gpu::StreamAccessor::getStream(s);
    cu_symmetry<<<cblocks,cthreads,0,cs>>>(smag, sdir, mag, dir, sigma);
  }

  void 
  symmetry_transform(const cv::Mat& img, cv::Mat& sym_mag, 
		     cv::Mat& sym_dir, symmetry_state_t& sb)
  {
    assert(cv::gpu::getCudaEnabledDeviceCount() > 0);
    cv::gpu::Stream s;
    // upload img to GPU
    sb.graw.upload(img);
    s.enqueueConvert(sb.graw, sb.gimg, CV_32F, 1.0/255.0);
    // compute the derivatives
    cv::gpu::createContinuous(img.size(), CV_32F, sb.gdx);
    cv::gpu::createContinuous(img.size(), CV_32F, sb.gdy);
    cv::Mat xkern = (cv::Mat_<float>(1,3) << -1, 0, 1);
    cv::Mat ykern = xkern.t();
    cv::gpu::filter2D(sb.gimg, sb.gdx, -1, xkern, cv::Point(-1,-1), cv::BORDER_DEFAULT, s);
    cv::gpu::filter2D(sb.gimg, sb.gdy, -1, ykern, cv::Point(-1,-1), cv::BORDER_DEFAULT, s);
    // compute the gradient magnitude and direction
    gradient(sb.gdx, sb.gdy, sb.gmag, sb.gdir, s);
    // compute the symmetry for each pixel
    symmetry(sb.gsmag, sb.gsdir, sb.gmag, sb.gdir, sb.sigma, s);
    s.waitForCompletion();
    sb.gsmag.download(sym_mag);
    sb.gsdir.download(sym_dir);    
  }
}

