#ifndef MACRO_HPP_
#define MACRO_HPP_

/// @file macros.hpp @brief various project macros

/** The power factor in implicit function equations.
    The formula is:
    f = (1 - (distance/radius)^2)^ALPHA
 */
#define ALPHA 4

#define POTENTIAL_ISO (0.5f)
//#define LARGE_SUPPORT_PRIMITIVES

/// The size of a block in 2D CUDA kernels
#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8

/// Sets the amount of multisampling in X and Y
// FIXME: multisampling only works for x = y = 1 ...
// everything is broken since the adds of progressive mode
#define MULTISAMPX 1
#define MULTISAMPY 1

#endif // MACRO_HPP_
