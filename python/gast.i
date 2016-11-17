%module pygast

%{
#define SWIG_FILE_WITH_INIT
#include <gast/pygast.hpp>
%}

// handle numpy
%include numpy.i
%init %{
  import_array();
%}

%apply (unsigned char* IN_ARRAY2, int DIM1, int DIM2) {(unsigned char* img, int irows, int icols)}
%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float* sym, int srows, int scols),(float* dir, int drows, int dcols)} 

%include <gast/pygast.hpp>
