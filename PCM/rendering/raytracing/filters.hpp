#ifndef FILTERS_HPP__
#define FILTERS_HPP__


// -----------------------------------------------------------------------------

/// Transform an image buffer of float4 to an image buffer of int.
/// optionnaly filter the input buffer
/// @param do_filter Filter the raytracing buffer with an additive scheme.
/// negative pixels are ignored and every pixels inside the window defined
/// by filter_size are added.
/// @param filter_size Width and height of the window use to filter the image.
/// @param do_dither : dither the final result or not
__global__
void flatten_image(const float4* in_color,
                   const float*  in_depth,
                   int*      out_rgb24,
                   unsigned* out_depth,
                   int width,
                   int height,
                   bool do_filter,
                   int filter_size,
                   bool do_dither);

// -----------------------------------------------------------------------------

/// Fill device mem in 'color' and 'depth' with cl_color
void clean_pbos(int* color,
                unsigned* depth,
                int width,
                int height,
                float4 cl_color);

void clean_buffers(float4* d_buff_,
                   float* d_depth_,
                   float far_,
                   int width_,
                   int height_);

// -----------------------------------------------------------------------------

/// Compute the bloom effect in place with the buffer "d_rendu" of size
/// width*height
void do_bloom(float4* d_rendu,
              float4* d_img_buff, float4* d_bloom_buff,
              int width, int height);

#endif // FILTERS_HPP__
