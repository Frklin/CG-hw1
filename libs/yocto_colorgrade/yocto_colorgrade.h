//
// Yocto/Grade: Tiny library for color grading.
//

//
// LICENSE:
//
// Copyright (c) 2020 -- 2020 Fabio Pellacini
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#ifndef _YOCTO_COLORGRADE_
#define _YOCTO_COLORGRADE_

#include <yocto/yocto_image.h>
#include <yocto/yocto_math.h>

// -----------------------------------------------------------------------------
// COLOR GRADING FUNCTIONS
// -----------------------------------------------------------------------------
namespace yocto {

// Color grading parameters
struct grade_params {
  float exposure   = 0.0f;
  bool  filmic     = false;
  bool  sepia      = false;
  bool  invert      = false;
  bool  srgb       = true;
  vec3f tint       = vec3f{1, 1, 1};
  float saturation = 0.5f;
  float contrast   = 0.5f;
  float vignette   = 0.0f;
  float grain      = 0.0f;
  int   mosaic     = 0;
  int   grid       = 0;
  vec2f sigma      = vec2f{0.0f, 0.0f};
  vec3f laplace     = vec3f{0.0f, 0.0f, 0.0f};
  bool  grayscale   = false;
};

// Grading functions
color_image grade_image(const color_image& image, const grade_params& params);


color_image gaussian_blur(color_image graded, float sigmax, float sigmay);
color_image local_laplace_filters(const color_image& image, const grade_params& params);



//gassuian blur
const vec4f        GetPixelOrBlack(const color_image& image, int x, int y);
std::vector<float> GaussianKernelIntegrals(float sigma, int taps);
float              Gaussian(float sigma, float x);
float              GaussianSimpsonIntegration(float sigma, float a, float b);

//gaussian pyr
double WeightingFunction(int i, double a);
void   populateImg(
      vector<color_image>& pyramid, int rowoff, int coloff, double alpha);
vector<color_image> constructGaussianPyramid(
    int levels, const grade_params& params, const color_image& image);
vec4i       GetLevelSize(int level, vector<color_image>& pyramid);
color_image Expand(vector<color_image>& gauss, int level, int times,
    const grade_params& params);
void        Expand(const color_image& input, int row_offset, int col_offset,
           color_image& output, const grade_params& params);

int GetLevelCount(int rows, int cols, int desired_base_size);

//laplace pyr

 vector<color_image> constructLaplacePyramid(
    int levels, const grade_params& params, vector<color_image>& gauss);
color_image Reconstruct(
    const grade_params& params, vector<color_image>& pyramid);
 
 
 //remapping
vec4f Evaluate(
    const vec4f value, const vec4f reference, const grade_params& params);
color_image Evaluate(const color_image& input, const vec4f& reference,
    const grade_params& params);
 vec4f Evaluate(double input, double reference, const grade_params& params);
double SmoothStep(double x_min, double x_max, double x);
double DetailRemap(double delta, const grade_params& params);
double      EdgeRemap(double delta, double beta);

//utility

color_image subregion(
    vec2i row, vec2i col, const color_image& image, const grade_params& params);
color_image diff(const color_image& i1, const color_image& i2);
color_image summ(color_image i1, color_image i2);
color_image grayscale(const color_image& image);
};  // namespace yocto


#endif
