#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <gpu_buffers/gpu_buffers.hpp>

namespace gast {

  struct symmetry_state_t 
  {
    int sigma;
    cv::gpu::GpuMat graw;
    cv::gpu::GpuMat gimg;
    cv::gpu::GpuMat gdx;
    cv::gpu::GpuMat gdy;
    cv::gpu::GpuMat gmag;
    cv::gpu::GpuMat gdir;
    cv::gpu::GpuMat gsmag;
    cv::gpu::GpuMat gsdir;
  };

  /*
    The symmetry transform expects a grayscale image and outputs a
    symmetry map of the same width and height as the input, where each
    element represents the strength of the symmetry at that point. 
   */

  void symmetry_transform(const cv::Mat& img, cv::Mat& sym_mag, 
			  cv::Mat& sym_dir, symmetry_state_t& sb);
  void symmetry_transform(const cv::gpu::GpuMat& img, cv::gpu::GpuMat& sym_map, 
			  symmetry_state_t& sb);
  void symmetry_transform(const gpu_buffers::GPUBuffer<uint8_t>& buf, 
			  gpu_buffers::GPUBuffer<float>& sym_mag, 
			  gpu_buffers::GPUBuffer<float>& sym_dir, 
			  symmetry_state_t& sb);

}

