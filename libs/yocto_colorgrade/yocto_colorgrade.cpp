//
// Implementation for Yocto/Grade.
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

#include "yocto_colorgrade.h"

#include <yocto/yocto_color.h>
#include <yocto/yocto_sampling.h>
// -----------------------------------------------------------------------------
// COLOR GRADING FUNCTIONS
// -----------------------------------------------------------------------------
namespace yocto {


color_image grade_image(const color_image& image, const grade_params& params) {
  auto graded = image;
  auto rng    = make_rng(0.0, 1.0);

  for (auto j : range(image.height))
    for (auto i : range(image.width)) {

      auto c = image[{i, j}];
      // Tone Mapping
      c *= pow(2, params.exposure);
       if (params.filmic) {
        c *= 0.6;
        c = (c * c * 2.51 + c * 0.03) / (c * c * 2.43 + c * 0.59 + 0.14);
      }
       if (params.srgb)
         c = pow(c, 1 / 2.2);
      c = clamp(c, 0, 1);
      // Color Tint
       c.x *= params.tint.x;
      c.y *= params.tint.y;
      c.z *= params.tint.z;
      // Saturation
       auto g = (c.x + c.y + c.z) / 3;
      c      = g + (c - g) * (params.saturation * 2);
      // Contrast
       c = gain(c, 1 - params.contrast);
      // Vignette
      vec2f ij   = {i, j};
      vec2f size = {image.width, image.height};
      auto  vr   = 1 - params.vignette;
      auto  r    = length(ij - size / 2) / length(size / 2);
      c *= (1 - smoothstep(vr, 2 * vr, r));
      // Film Grain
       c += (rand1f(rng) - 0.5) * params.grain;
      // Sepia
       if (params.sepia) {
         int tr = 0.393 * float_to_byte(c.x) + 0.769 * float_to_byte(c.y) +
                  0.189 * float_to_byte(c.z);
         int tg = 0.349 * float_to_byte(c.x) + 0.686 * float_to_byte(c.y) +
                  0.168 * float_to_byte(c.z);
         int tb = 0.272 * float_to_byte(c.x) + 0.534 * float_to_byte(c.y) +
                  0.131 * float_to_byte(c.z);
         c.x = (tr > 255) ? 255 : byte_to_float(tr);
         c.y = (tg > 255) ? 255 : byte_to_float(tg);
         c.z = (tb > 255) ? 255 : byte_to_float(tb);
       }


       // Saving Image
       set_pixel(graded, i, j, {c.x, c.y, c.z, image[{i, j}].w});
    }

  if (params.grayscale)
    graded = grayscale(graded);
  if (params.sigma.x != 0 || params.sigma.y != 0) {
    auto blurred = gaussian_blur(graded, params.sigma.x, params.sigma.y);
    graded = blurred;
  }

  
  if (params.laplace.x != 0 || params.laplace.y != 0 || params.laplace.z != 0) {
    auto img = local_laplace_filters(graded, params);
    graded = img;
  }
  if (params.invert) {
    auto inverted= graded;
    for (auto j : range(graded.height))
      for (auto i : range(graded.width)) {
        auto c = graded[{i, j}];
        c         = 1 - c;
        set_pixel(inverted, i, j, c);
      }
    graded = inverted;
  }

    if (params.mosaic != 0) {
    auto mosaic = graded;
    for (auto j : range(graded.height))
      for (auto i : range(graded.width)) {
        auto c = graded[{i, j}];
        // Mosaic Effect
        c = graded[{i - (i % params.mosaic), j - (j % params.mosaic)}];
        // Grid Effect
        if (params.grid != 0)
          c = (i % params.grid == 0 || 0 == j % params.grid) ? 0.5 * c : c;
        set_pixel(mosaic, i, j, {c.x, c.y, c.z, image[{i, j}].w});
      }
    graded = mosaic;
  }

  return graded;
}
color_image gaussian_blur(color_image graded, float sigmax, float sigmay) {
  auto  tmp       = graded;
  float threshold = 0.005f;
  int   xblursize = floor(
      1.0f + 2.0f * sqrt(-2.0f * pow(sigmax, 2) * log(threshold)));
  int yblursize = floor(
      1.0f + 2.0f * sqrt(-2.0f * pow(sigmay, 2) * log(threshold)));
  auto rowx         = GaussianKernelIntegrals(sigmax, xblursize);
  auto rowy         = GaussianKernelIntegrals(sigmay, yblursize);
  int  startOffset = -1 * int(rowx.size() / 2);
  if (sigmax!=0)
  for (int y = 0; y < graded.height; ++y)
    for (int x = 0; x < graded.width; ++x) {
      auto  c            = graded[{x, y}];
      vec4f blurredpixel = {0.0f, 0.0f, 0.0f, graded[{x, y}].w};
      for (unsigned i = 0; i < rowx.size(); ++i) {
        const vec4f pixel = GetPixelOrBlack(graded, x + startOffset + i, y);
        blurredpixel.x += float_to_byte(pixel.x) * rowx[i];
        blurredpixel.y += float_to_byte(pixel.y) * rowx[i];
        blurredpixel.z += float_to_byte(pixel.z) * rowx[i];
      }
      c.x = byte_to_float(blurredpixel.x);
      c.y = byte_to_float(blurredpixel.y);
      c.z = byte_to_float(blurredpixel.z);
      set_pixel(tmp, x, y, c);
    }

  auto blurred = tmp;
  
  startOffset = -1 * int(rowy.size() / 2);
  if (sigmay!=0)
  for (int y = 0; y < tmp.height; ++y)
    for (int x = 0; x < tmp.width; ++x) {
      auto  c            = tmp[{x, y}];
      vec4f blurredpixel = {0.0f, 0.0f, 0.0f, graded[{x, y}].w};
      for (unsigned i = 0; i < rowy.size(); ++i) {
        const vec4f pixel = GetPixelOrBlack(tmp, x, y + startOffset + i);
        blurredpixel.x += float_to_byte(pixel.x) * rowy[i];
        blurredpixel.y += float_to_byte(pixel.y) * rowy[i];
        blurredpixel.z += float_to_byte(pixel.z) * rowy[i];
      }
      c.x = byte_to_float(blurredpixel.x);
      c.y = byte_to_float(blurredpixel.y);
      c.z = byte_to_float(blurredpixel.z);
      set_pixel(blurred, x, y, c);
    }
  return blurred;
}




color_image local_laplace_filters(
    const color_image& image, const grade_params& params) {
  auto graded = image;
  int  full_res_y, roi_y0, roi_y1, full_res_roi_y, full_res_x, roi_x0, roi_x1,
      full_res_roi_x;
  vec2i row_range, col_range;

  int num_levels =GetLevelCount(image.height, image.width,50);  

  auto gausspyr = constructGaussianPyramid(num_levels, params,image);
  auto laplacepyr = 
      gausspyr;  

  laplacepyr[num_levels] = gausspyr[num_levels];

  for (int l = 0; l < num_levels; l++) {
    int subregion_size = 3 * ((1 << (l + 2)) - 1);
    int subregion_r    = subregion_size / 2;

    for (int y = 0; y < laplacepyr[l].height; y++) {
        full_res_y = (1 << l) * y;
        roi_y0     = full_res_y - subregion_r;
        roi_y1     = full_res_y + subregion_r + 1;
        row_range   ={max(0, roi_y0), min(roi_y1, image.height)};
        full_res_roi_y = full_res_y - row_range.x;
      for (int x = 0; x < laplacepyr[l].width; x++) {
        full_res_x     = (1 << l) * x;
        roi_x0         = full_res_x - subregion_r;
        roi_x1         = full_res_x + subregion_r + 1;
        col_range      = {max(0, roi_x0), min(roi_x1, image.width)};
        full_res_roi_x = full_res_x - col_range.x;

        auto g0 = gausspyr[l][{x, y}];

        auto r0 = subregion(row_range, col_range, image, params);

        auto remapped = Evaluate(r0,  g0, params); 
        
        auto tmp      = constructLaplacePyramid(
            l + 1, params, constructGaussianPyramid(l + 1, params, remapped));

        laplacepyr[l][{x, y}] = tmp[l][{full_res_roi_x >> l, full_res_roi_y >> l}];

      }
    }
    printf("Livello %d/%d costruito\n", l+1,num_levels);
  }

    return Reconstruct(params,laplacepyr);
}

//gaussian blur
float Gaussian(float sigma, float x) {
  return exp(-(x * x) / (2.0f * sigma * sigma));  // expf?
}
float GaussianSimpsonIntegration(float sigma, float a, float b) {
  return ((b - a) / 6.0f) *
         (Gaussian(sigma, a) + 4.0f * Gaussian(sigma, (a + b) / 2.0f) +
             Gaussian(sigma, b));
}
std::vector<float> GaussianKernelIntegrals(float sigma, int taps) {
  std::vector<float> ret;
  float              total = 0.0f;
  for (int i = 0; i < taps; ++i) {
    float x     = float(i) - float(taps / 2);
    float value = GaussianSimpsonIntegration(sigma, x - 0.5f, x + 0.5f);
    ret.push_back(value);
    total += value;
  }
  // normalize it
  for (unsigned int i = 0; i < ret.size(); i++) {
    ret[i] /= total;
  }
  return ret;
}
const vec4f GetPixelOrBlack(const color_image& image, int x, int y) {
  const vec4f black = {0, 0, 0, 0};
  if (x < 0 || x >= image.width || y < 0 || y >= image.height) {
    return black;
  }
  return image[{x, y}];
}

//gausspyr
 vector<color_image> constructGaussianPyramid(
    int levels, const grade_params& params, const color_image& image) {
  vector<color_image> pyramid;
   pyramid.emplace_back(image);
  for (int i = 0; i<levels;i++) {
    auto oldImg = pyramid.back();
    vec4i oldsubwind = GetLevelSize(i,pyramid);
    vec4i newsubwind = GetLevelSize(i+1,pyramid);
    int   newHeight  = newsubwind.y - newsubwind.x + 1;
    int   newWidth  = newsubwind.w - newsubwind.z + 1;
    
    auto newImg = make_image(newWidth, newHeight, params.srgb);
    pyramid.emplace_back(newImg);

    int rowOff = ((oldsubwind.x % 2) == 0) ? 0 : 1;
    int colOff = ((oldsubwind.z % 2) == 0) ? 0 : 1;

    populateImg(pyramid,rowOff,colOff,params.laplace.z);
    //return pyramid;
  }
  return pyramid;
 }
void populateImg(vector<color_image>& pyramid,int rowoff, int coloff,double alpha) {
  //int  c      = 0;
  auto oldImg = pyramid[pyramid.size() - 2];
  auto newImg = pyramid.back();

  for (int y = rowoff; y < oldImg.height; y+= 2) 
    for (int x = coloff; x < oldImg.width; x += 2) {
      vec4f value        = {0, 0, 0, 0};
      double total_weight = 0;
      int row_start = std::max(0, y - 2);           //0
      int    row_end      = std::min(oldImg.height - 1, y + 2);  //2
      int    col_start    = std::max(0, x - 2);                  //1078
      int    col_end      = std::min(oldImg.width - 1, x + 2);   //1082  
      for (int n = row_start; n <= row_end; n++) {
        double row_weight = WeightingFunction(n - y, alpha);
        for (int m = col_start; m <= col_end; m++) {
         // printf("y = %d x = %d n = %d m =%d \n",y,x, n, m);
          double weight = row_weight * WeightingFunction(m - x, alpha);
          total_weight += weight;
          value += weight * oldImg[{m, n}];  // soli pixel?
        }
      }
      set_pixel(newImg, x >> 1, y >> 1,value / total_weight);
    }
  pyramid[pyramid.size() - 1] = newImg;
}
double WeightingFunction(int i, double a) {
  switch (i) {
    case 0: return 0.4;
    case -1:
    case 1: return 0.25;
    case -2:
    case 2: return 0.25 - 0.5 * 0.4;
  }
  return 0;
}
vec4i GetLevelSize(int level, vector<color_image>& pyramid) {
  vec4i sub = {0, pyramid[0].height-1, 0, pyramid[0].width-1};
  for (int i = 0; i < level; i++) {
    sub.x = (sub.x >> 1) + sub.x % 2;
    sub.y = (sub.y >> 1);
    sub.z = (sub.z >> 1) + sub.z % 2;
    sub.w = (sub.w >> 1);
  }
  return sub;
  // GetLevelSize(subwindow_, level, subwindow);
}
color_image Expand(
    vector<color_image>& gauss,int level, int times, const grade_params& params) {
  if (times < 1) return gauss[level];
  times = min(times, level);

  color_image base = gauss[level], expanded;

  for (int i = 0; i < times; i++) {
    vec4i subwindow = GetLevelSize(level - i - 1,gauss);  
    expanded     = make_image(gauss[level - i - 1].width,
        gauss[level - i - 1].height, params.srgb);  

    Expand(base, ((subwindow.x % 2) == 0) ? 0 : 1,
        ((subwindow.z % 2) == 0) ? 0 : 1, expanded, params);

    base = expanded;
  }

  return expanded;
}
void Expand(const color_image& input, int row_offset, int col_offset,
    color_image& output, const grade_params& params) {
  color_image upsamp = make_image(output.width, output.height, params.srgb);    
  std::vector<int> norm;
  norm.resize(output.height * output.width);

  for (int i = row_offset; i < output.height; i += 2) {
    for (int j = col_offset; j < output.width; j += 2) {
        set_pixel(upsamp, j, i, input[{j >> 1, i >> 1}]);
        norm[i*output.width+ j] = 1;
      }
  }

  double filter[5][5];
  for (int i = -2; i <= 2; i++) {
    for (int j = -2; j <= 2; j++) {
      filter[i + 2][j + 2] = WeightingFunction(i,params.laplace.z ) *
                             WeightingFunction(j, params.laplace.z);
    }
  }

  for (int i = 0; i < output.height; i++) {
    int row_start = max(0, i - 2);
    int row_end   = min(output.height - 1, i + 2);
    for (int j = 0; j < output.width; j++) {
      int    col_start    = max(0, j - 2);
      int    col_end      = min(output.width - 1, j + 2);
      vec4f  value        = {0, 0, 0, 0};
      double total_weight = 0;
      for (int n = row_start; n <= row_end; n++) {
        for (int m = col_start; m <= col_end; m++) {
          double weight = filter[n - i + 2][m - j + 2];
          value += weight * upsamp[{m, n}];
          total_weight += weight * norm[n*output.width+m];
        }
      }
      set_pixel(output, j, i, value / total_weight);
    }
  }
}

 //laplacian pyr
vector<color_image> constructLaplacePyramid(
    int levels, const grade_params& params, vector<color_image>& gauss) {
  vector<color_image> laplace;
  for (int i = 0; i < levels; i++) {
    laplace.emplace_back(diff(gauss[i], Expand(gauss, i + 1, 1, params)));
  }
  laplace.emplace_back(gauss[levels]);
  return laplace;
}
color_image Reconstruct(
    const grade_params& params, vector<color_image>& pyramid) {
  auto        base = pyramid.back();
  color_image expanded;
  vec4i       subwindow;

  for (int i = pyramid.size() - 2; i >= 0; i--) {
    subwindow = GetLevelSize(i, pyramid);

    int row_offset = ((subwindow.x % 2) == 0) ? 0 : 1;
    int col_offset = ((subwindow.z % 2) == 0) ? 0 : 1;

    expanded = make_image(pyramid[i].width, pyramid[i].height, params.srgb);
    Expand(base, row_offset, col_offset, expanded, params);
    base = summ(expanded, pyramid[i]);
  }

  return base;
}




//remapping function
color_image Evaluate(const color_image& input, const vec4f& reference,
    const grade_params& params) {
  auto output = make_image(input.width, input.height, params.srgb);
  for (int i = 0; i < input.height; i++) {
    for (int j = 0; j < input.width; j++) {
      auto pixel = input[{j, i}];
      if (params.grayscale)
        pixel = Evaluate(input[{j, i}].x, reference.x, params);
      else
        pixel = Evaluate(input[{j, i}], reference, params);
      output[{j, i}] = pixel;
    }
  }
  return output;
}
vec4f Evaluate(double input, double reference, const grade_params& params) {
  auto  diff = input - reference;
  auto  sign = diff > 0 ? 1 : -1;
  float val;
  if (diff <= params.laplace.z) {
    val = reference +
          sign * params.laplace.z * DetailRemap(::abs(diff), params);
  } else {
    val = reference +
          sign * (params.laplace.y * (::abs(diff) - params.laplace.z) +
                     params.laplace.z);
  }
  return {val, val, val, 1.0f};
}
vec4f Evaluate(
    const vec4f value, const vec4f reference, const grade_params& params) {
  auto   delta = value - reference;
  double mag   = ::sqrt(
      delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
  if (mag > 1e-10) delta /= mag;

  if (mag < params.laplace.z) {
    return value + delta * params.laplace.z * DetailRemap(mag, params);
  } else {
    return value +
           delta * (EdgeRemap(mag - params.laplace.z, params.laplace.y) +
                       params.laplace.z);
  }
}
double EdgeRemap(double delta, double beta) { return delta * beta; }
double DetailRemap(double delta, const grade_params& params) {
  double fraction   = delta / params.laplace.z;
  double polynomial = std::pow(fraction, params.laplace.x);
  if (params.laplace.x < 1) {
    const double kNoiseLevel = 0.01;
    double       blend       = SmoothStep(
        kNoiseLevel, 2 * kNoiseLevel, fraction * params.laplace.z);
    polynomial = blend * polynomial + (1 - blend) * fraction;
  }
  return polynomial;
}
double SmoothStep(double x_min, double x_max, double x) {
  double y = (x - x_min) / (x_max - x_min);
  y        = max(0.0, min(1.0, y));
  return pow(y, 2) * pow(y - 2, 2);
}
//utlity
int GetLevelCount(int rows, int cols, int desired_base_size) {
  int min_dim = std::min(rows, cols);

  double log2_dim = std::log2(min_dim);
  double log2_des = std::log2(desired_base_size);

  return static_cast<int>(std::ceil(std::abs(log2_dim - log2_des)));
}
color_image grayscale(const color_image& image) {
  auto img = image;
  for (auto& px : img.pixels) {
    auto val = (px.x + px.y + px.z) / 3;
    px.x     = val;
    px.y     = val;
    px.z     = val;
  }
  return img;
}
color_image diff(const color_image& i1, const color_image& i2) {
  auto imm = i1;
  for (auto j : range(i1.height))
    for (auto i : range(i2.width)) {
      auto pixel = i1[{i, j}];
      pixel      = i1[{i, j}] - i2[{i, j}];
      set_pixel(imm, i, j, pixel);
    }
  return imm;
}
color_image summ(color_image i1, color_image i2) {
  auto imm = i1;
  for (auto j : range(i1.height))
    for (auto i : range(i2.width)) {
      auto pixel = i1[{i, j}];
      pixel      = i1[{i, j}] + i2[{i, j}];
      set_pixel(imm, i, j, pixel);
    }
  return imm;
}
color_image subregion(vec2i row, vec2i col, const color_image& image,
    const grade_params& params) {
  auto dest = make_image(col.y - col.x, row.y - row.x, params.srgb);
  for (int j = 0; j < dest.height; j++)
    for (int i = 0; i < dest.width; i++)
      set_pixel(dest, i, j, image[{col.x + i, row.x + j}]);
  return dest;
}

}  // namespace yocto