#pragma once

struct symmetry_impl_t;

class Symmetry {
public:
  Symmetry();
  virtual ~Symmetry();

  void transform(unsigned char* img, int irows, int icols,
		 float* sym, int srows, int scols, 
		 float* dir, int drows, int dcols, 
		 int sigma = 30);
private:
  symmetry_impl_t* state_;
};