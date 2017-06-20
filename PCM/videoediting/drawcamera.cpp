#include "drawcamera.h"
#include "glut.h"
#include"toolbox\gl_utils\glassert.h"

// vertices for camera
static GLfloat cameraVertices[] = {
	0.500000f, -0.350000f, 0.000000f, 0.500000f, -0.350000f, 0.000000f, 0.500000f, -0.350000f, 0.000000f,
	-0.500000f, -0.350000f, 0.000000f, -0.500000f, -0.350000f, 0.000000f,
	-0.500000f, -0.350000f, 0.000000f, -0.500000f, 0.350000f, 0.000000f,
	-0.500000f, 0.350000f, 0.000000f, -0.500000f, 0.350000f, 0.000000f,
	0.500000f, 0.350000f, 0.000000f, 0.500000f, 0.350000f, 0.000000f,
	0.500000f, 0.350000f, 0.000000f, -0.500000f, 0.350000f, 0.300000f,
	-0.500000f, 0.350000f, 0.300000f, -0.500000f, 0.350000f, 0.300000f,
	0.500000f, 0.350000f, 0.300000f, 0.500000f, 0.350000f, 0.300000f,
	0.500000f, 0.350000f, 0.300000f, -0.500000f, -0.350000f, 0.300000f,
	-0.500000f, -0.350000f, 0.300000f, -0.500000f, -0.350000f, 0.300000f,
	0.500000f, -0.350000f, 0.300000f, 0.500000f, -0.350000f, 0.300000f,
	0.500000f, -0.350000f, 0.300000f, -0.285317f, 0.0927050f, 0.000000f,
	-0.242705f, 0.176336f, 0.000000f, -0.242705f, 0.176336f, -0.300000f,
	-0.242705f, 0.176336f, -0.300000f, -0.285317f, 0.0927050f, -0.300000f,
	-0.285317f, 0.0927050f, -0.300000f, -0.176336f, 0.242705f, 0.000000f,
	-0.176336f, 0.242705f, -0.300000f, -0.176336f, 0.242705f, -0.300000f,
	-0.0927050f, 0.285317f, 0.000000f, -0.0927050f, 0.285317f, -0.300000f,
	-0.0927050f, 0.285317f, -0.300000f, 0.000000f, 0.300000f, 0.000000f,
	0.000000f, 0.300000f, -0.300000f, 0.000000f, 0.300000f, -0.300000f,
	0.0927050f, 0.285317f, 0.000000f, 0.0927050f, 0.285317f, -0.300000f,
	0.0927050f, 0.285317f, -0.300000f, 0.176336f, 0.242705f, 0.000000f,
	0.176336f, 0.242705f, -0.300000f, 0.176336f, 0.242705f, -0.300000f,
	0.242705f, 0.176336f, 0.000000f, 0.242705f, 0.176336f, -0.300000f,
	0.242705f, 0.176336f, -0.300000f, 0.285317f, 0.0927050f, 0.000000f,
	0.285317f, 0.0927050f, -0.300000f, 0.285317f, 0.0927050f, -0.300000f,
	0.300000f, 0.000000f, 0.000000f, 0.300000f, 0.000000f, -0.300000f,
	0.300000f, 0.000000f, -0.300000f, 0.285317f, -0.0927050f, 0.000000f,
	0.285317f, -0.0927050f, -0.300000f, 0.285317f, -0.0927050f, -0.300000f,
	0.242705f, -0.176336f, 0.000000f, 0.242705f, -0.176336f, -0.300000f,
	0.242705f, -0.176336f, -0.300000f, 0.176336f, -0.242705f, 0.000000f,
	0.176336f, -0.242705f, -0.300000f, 0.176336f, -0.242705f, -0.300000f,
	0.0927050f, -0.285317f, 0.000000f, 0.0927050f, -0.285317f, -0.300000f,
	0.0927050f, -0.285317f, -0.300000f, 0.000000f, -0.300000f, 0.000000f,
	0.000000f, -0.300000f, -0.300000f, 0.000000f, -0.300000f, -0.300000f,
	-0.0927050f, -0.285317f, 0.000000f, -0.0927050f, -0.285317f, -0.300000f,
	-0.0927050f, -0.285317f, -0.300000f, -0.176336f, -0.242705f, 0.000000f,
	-0.176336f, -0.242705f, -0.300000f, -0.176336f, -0.242705f, -0.300000f,
	-0.242705f, -0.176336f, 0.000000f, -0.242705f, -0.176336f, -0.300000f,
	-0.242705f, -0.176336f, -0.300000f, -0.285317f, -0.0927050f, 0.000000f,
	-0.285317f, -0.0927050f, -0.300000f, -0.285317f, -0.0927050f, -0.300000f,
	-0.300000f, 0.000000f, 0.000000f, -0.300000f, 0.000000f, -0.300000f,
	-0.300000f, 0.000000f, -0.300000f, -0.194164f, 0.141069f, -0.300000f,
	-0.194164f, 0.141069f, -0.300000f, -0.228254f, 0.0741640f, -0.300000f,
	-0.228254f, 0.0741640f, -0.300000f, -0.141069f, 0.194164f, -0.300000f,
	-0.141069f, 0.194164f, -0.300000f, -0.0741640f, 0.228254f, -0.300000f,
	-0.0741640f, 0.228254f, -0.300000f, 0.000000f, 0.240000f, -0.300000f,
	0.000000f, 0.240000f, -0.300000f, 0.0741640f, 0.228254f, -0.300000f,
	0.0741640f, 0.228254f, -0.300000f, 0.141069f, 0.194164f, -0.300000f,
	0.141069f, 0.194164f, -0.300000f, 0.194164f, 0.141069f, -0.300000f,
	0.194164f, 0.141069f, -0.300000f, 0.228254f, 0.0741640f, -0.300000f,
	0.228254f, 0.0741640f, -0.300000f, 0.240000f, 0.000000f, -0.300000f,
	0.240000f, 0.000000f, -0.300000f, 0.228254f, -0.0741640f, -0.300000f,
	0.228254f, -0.0741640f, -0.300000f, 0.194164f, -0.141069f, -0.300000f,
	0.194164f, -0.141069f, -0.300000f, 0.141069f, -0.194164f, -0.300000f,
	0.141069f, -0.194164f, -0.300000f, 0.0741640f, -0.228254f, -0.300000f,
	0.0741640f, -0.228254f, -0.300000f, 0.000000f, -0.240000f, -0.300000f,
	0.000000f, -0.240000f, -0.300000f, -0.0741640f, -0.228254f, -0.300000f,
	-0.0741640f, -0.228254f, -0.300000f, -0.141068f, -0.194164f, -0.300000f,
	-0.141068f, -0.194164f, -0.300000f, -0.194164f, -0.141068f, -0.300000f,
	-0.194164f, -0.141068f, -0.300000f, -0.228254f, -0.0741640f, -0.300000f,
	-0.228254f, -0.0741640f, -0.300000f, -0.240000f, 0.000000f, -0.300000f,
	-0.240000f, 0.000000f, -0.300000f, -0.228254f, 0.0741640f, 0.000000f,
	-0.194164f, 0.141069f, 0.000000f, -0.141069f, 0.194164f, 0.000000f,
	-0.0741640f, 0.228254f, 0.000000f, 0.000000f, 0.240000f, 0.000000f,
	0.0741640f, 0.228254f, 0.000000f, 0.141069f, 0.194164f, 0.000000f,
	0.194164f, 0.141069f, 0.000000f, 0.228254f, 0.0741640f, 0.000000f,
	0.240000f, 0.000000f, 0.000000f, 0.228254f, -0.0741640f, 0.000000f,
	0.194164f, -0.141069f, 0.000000f, 0.141069f, -0.194164f, 0.000000f,
	0.0741640f, -0.228254f, 0.000000f, 0.000000f, -0.240000f, 0.000000f,
	-0.0741640f, -0.228254f, 0.000000f, -0.141068f, -0.194164f, 0.000000f,
	-0.194164f, -0.141068f, 0.000000f, -0.228254f, -0.0741640f, 0.000000f,
	-0.240000f, 0.000000f, 0.000000f, 0.306365f, 0.350000f, 0.164697f,
	0.313467f, 0.350000f, 0.178636f, 0.313467f, 0.370000f, 0.178636f,
	0.313467f, 0.370000f, 0.178636f, 0.306365f, 0.370000f, 0.164697f,
	0.306365f, 0.370000f, 0.164697f, 0.324529f, 0.350000f, 0.189697f,
	0.324529f, 0.370000f, 0.189697f, 0.324529f, 0.370000f, 0.189697f,
	0.338467f, 0.350000f, 0.196799f, 0.338467f, 0.370000f, 0.196799f,
	0.338467f, 0.370000f, 0.196799f, 0.353918f, 0.350000f, 0.199246f,
	0.353918f, 0.370000f, 0.199246f, 0.353918f, 0.370000f, 0.199246f,
	0.369369f, 0.350000f, 0.196799f, 0.369369f, 0.370000f, 0.196799f,
	0.369369f, 0.370000f, 0.196799f, 0.383307f, 0.350000f, 0.189697f,
	0.383307f, 0.370000f, 0.189697f, 0.383307f, 0.370000f, 0.189697f,
	0.394369f, 0.350000f, 0.178636f, 0.394369f, 0.370000f, 0.178636f,
	0.394369f, 0.370000f, 0.178636f, 0.401471f, 0.350000f, 0.164697f,
	0.401471f, 0.370000f, 0.164697f, 0.401471f, 0.370000f, 0.164697f,
	0.403918f, 0.350000f, 0.149246f, 0.403918f, 0.370000f, 0.149246f,
	0.403918f, 0.370000f, 0.149246f, 0.401471f, 0.350000f, 0.133795f,
	0.401471f, 0.370000f, 0.133795f, 0.401471f, 0.370000f, 0.133795f,
	0.394369f, 0.350000f, 0.119857f, 0.394369f, 0.370000f, 0.119857f,
	0.394369f, 0.370000f, 0.119857f, 0.383307f, 0.350000f, 0.108795f,
	0.383307f, 0.370000f, 0.108795f, 0.383307f, 0.370000f, 0.108795f,
	0.369369f, 0.350000f, 0.101693f, 0.369369f, 0.370000f, 0.101693f,
	0.369369f, 0.370000f, 0.101693f, 0.353918f, 0.350000f, 0.0992460f,
	0.353918f, 0.370000f, 0.0992460f, 0.353918f, 0.370000f, 0.0992460f,
	0.338467f, 0.350000f, 0.101693f, 0.338467f, 0.370000f, 0.101693f,
	0.338467f, 0.370000f, 0.101693f, 0.324529f, 0.350000f, 0.108795f,
	0.324529f, 0.370000f, 0.108795f, 0.324529f, 0.370000f, 0.108795f,
	0.313467f, 0.350000f, 0.119857f, 0.313467f, 0.370000f, 0.119857f,
	0.313467f, 0.370000f, 0.119857f, 0.306365f, 0.350000f, 0.133795f,
	0.306365f, 0.370000f, 0.133795f, 0.306365f, 0.370000f, 0.133795f,
	0.303918f, 0.350000f, 0.149246f, 0.303918f, 0.370000f, 0.149246f,
	0.303918f, 0.370000f, 0.149246f, 0.353918f, 0.370000f, 0.149246f
};
static GLfloat cameraNormals[] = {
	1.00000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-1.00000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, -1.00000f, 0.000000f, 0.000000f,
	0.000000f, 1.00000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	1.00000f, 0.000000f, 0.000000f, 0.000000f, 1.00000f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, -1.00000f, 0.000000f, 0.000000f,
	0.000000f, 0.000000f, 1.00000f, 0.000000f, 1.00000f, 0.000000f,
	1.00000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.00000f,
	0.000000f, 1.00000f, 0.000000f, -1.00000f, 0.000000f, 0.000000f,
	0.000000f, -1.00000f, 0.000000f, 0.000000f, 0.000000f, 1.00000f,
	1.00000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f, 0.000000f,
	0.000000f, 0.000000f, 1.00000f, -0.951057f, 0.309016f, 0.000000f,
	-0.809017f, 0.587785f, 0.000000f, -0.809017f, 0.587785f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, -0.951057f, 0.309016f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, -0.587785f, 0.809017f, 0.000000f,
	-0.587785f, 0.809017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.309017f, 0.951057f, 0.000000f, -0.309017f, 0.951057f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, 0.000000f, 1.00000f, 0.000000f,
	0.000000f, 1.00000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.309017f, 0.951056f, 0.000000f, 0.309017f, 0.951056f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, 0.587785f, 0.809017f, 0.000000f,
	0.587785f, 0.809017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.809017f, 0.587785f, 0.000000f, 0.809017f, 0.587785f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, 0.951057f, 0.309017f, 0.000000f,
	0.951057f, 0.309017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	1.00000f, 0.000000f, 0.000000f, 1.00000f, 0.000000f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, 0.951057f, -0.309017f, 0.000000f,
	0.951057f, -0.309017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.809017f, -0.587785f, 0.000000f, 0.809017f, -0.587785f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, 0.587785f, -0.809017f, 0.000000f,
	0.587785f, -0.809017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.309017f, -0.951057f, 0.000000f, 0.309017f, -0.951057f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, 0.000000f, -1.00000f, 0.000000f,
	0.000000f, -1.00000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.309017f, -0.951056f, 0.000000f, -0.309017f, -0.951056f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, -0.587785f, -0.809017f, 0.000000f,
	-0.587785f, -0.809017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.809017f, -0.587785f, 0.000000f, -0.809017f, -0.587785f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, -0.951057f, -0.309017f, 0.000000f,
	-0.951057f, -0.309017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-1.00000f, -1.00000e-006f, 0.000000f, -1.00000f, -1.00000e-006f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, 0.000000f, 0.000000f, -1.00000f,
	0.809017f, -0.587785f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.951057f, -0.309016f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.587785f, -0.809017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.309017f, -0.951056f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.000000f, -1.00000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.309017f, -0.951056f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.587785f, -0.809017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.809017f, -0.587785f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.951057f, -0.309017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-1.00000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.951057f, 0.309017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.809017f, 0.587785f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.587785f, 0.809017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.309017f, 0.951057f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.000000f, 1.00000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.309017f, 0.951056f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.587785f, 0.809017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.809017f, 0.587785f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.951057f, 0.309017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	1.00000f, 1.00000e-006f, 0.000000f, 0.951057f, -0.309016f, 0.000000f,
	0.809017f, -0.587785f, 0.000000f, 0.587785f, -0.809017f, 0.000000f,
	0.309017f, -0.951056f, 0.000000f, 0.000000f, -1.00000f, 0.000000f,
	-0.309017f, -0.951056f, 0.000000f, -0.587785f, -0.809017f, 0.000000f,
	-0.809017f, -0.587785f, 0.000000f, -0.951057f, -0.309017f, 0.000000f,
	-1.00000f, 0.000000f, 0.000000f, -0.951057f, 0.309017f, 0.000000f,
	-0.809017f, 0.587785f, 0.000000f, -0.587785f, 0.809017f, 0.000000f,
	-0.309017f, 0.951057f, 0.000000f, 0.000000f, 1.00000f, 0.000000f,
	0.309017f, 0.951056f, 0.000000f, 0.587785f, 0.809017f, 0.000000f,
	0.809017f, 0.587785f, 0.000000f, 0.951057f, 0.309017f, 0.000000f,
	1.00000f, 1.00000e-006f, 0.000000f, -0.951057f, 0.000000f, 0.309017f,
	-0.809017f, 0.000000f, 0.587786f, -0.809017f, 0.000000f, 0.587786f,
	0.000000f, 1.00000f, -1.00000e-006f, -0.951057f, 0.000000f, 0.309017f,
	0.000000f, 1.00000f, -2.00000e-006f, -0.587785f, 0.000000f, 0.809017f,
	-0.587785f, 0.000000f, 0.809017f, 0.000000f, 1.00000f, 0.000000f,
	-0.309016f, 0.000000f, 0.951057f, -0.309016f, 0.000000f, 0.951057f,
	0.000000f, 1.00000f, 0.000000f, 1.00000e-006f, 0.000000f, 1.00000f,
	1.00000e-006f, 0.000000f, 1.00000f, 0.000000f, 1.00000f, 0.000000f,
	0.309018f, 0.000000f, 0.951056f, 0.309018f, 0.000000f, 0.951056f,
	0.000000f, 1.00000f, 0.000000f, 0.587785f, 0.000000f, 0.809017f,
	0.587785f, 0.000000f, 0.809017f, 0.000000f, 1.00000f, 0.000000f,
	0.809017f, 0.000000f, 0.587786f, 0.809017f, 0.000000f, 0.587786f,
	0.000000f, 1.00000f, 0.000000f, 0.951057f, 0.000000f, 0.309017f,
	0.951057f, 0.000000f, 0.309017f, 0.000000f, 1.00000f, 1.00000e-006f,
	1.00000f, 0.000000f, 0.000000f, 1.00000f, 0.000000f, 0.000000f,
	0.000000f, 1.00000f, 2.00000e-006f, 0.951057f, 0.000000f, -0.309017f,
	0.951057f, 0.000000f, -0.309017f, 0.000000f, 1.00000f, 2.00000e-006f,
	0.809017f, 0.000000f, -0.587786f, 0.809017f, 0.000000f, -0.587786f,
	0.000000f, 1.00000f, 1.00000e-006f, 0.587785f, 0.000000f, -0.809017f,
	0.587785f, 0.000000f, -0.809017f, 0.000000f, 1.00000f, 0.000000f,
	0.309017f, 0.000000f, -0.951056f, 0.309017f, 0.000000f, -0.951056f,
	0.000000f, 1.00000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.000000f, 0.000000f, -1.00000f, 0.000000f, 1.00000f, 0.000000f,
	-0.309017f, 0.000000f, -0.951056f, -0.309017f, 0.000000f, -0.951056f,
	0.000000f, 1.00000f, 0.000000f, -0.587786f, 0.000000f, -0.809017f,
	-0.587786f, 0.000000f, -0.809017f, 0.000000f, 1.00000f, 0.000000f,
	-0.809017f, 0.000000f, -0.587785f, -0.809017f, 0.000000f, -0.587785f,
	0.000000f, 1.00000f, 0.000000f, -0.951056f, 0.000000f, -0.309018f,
	-0.951056f, 0.000000f, -0.309018f, 0.000000f, 1.00000f, -1.00000e-006f,
	-1.00000f, 0.000000f, -1.00000e-006f, -1.00000f, 0.000000f, -1.00000e-006f,
	0.000000f, 1.00000f, -2.00000e-006f, 0.000000f, 1.00000f, 0.000000f
};
static GLint cameraIndices[] = {
	2, 5, 11, 5, 8, 10, 7, 17, 7, 14, 16, 13, 23, 13, 20, 22,
	19, 1, 19, 4, 3, 18, 6, 18, 12, 21, 0, 15, 0, 9, 203, 149,
	204, 147, 204, 152, 204, 155, 204, 158, 204, 161, 204, 164, 204, 167, 204, 170,
	204, 173, 204, 176, 204, 179, 204, 182, 204, 185, 204, 188, 204, 191, 204, 194,
	204, 197, 204, 200, 203, 144, 148, 144, 202, 201, 199, 198, 196, 195, 193, 192,
	190, 189, 187, 186, 184, 183, 181, 180, 178, 177, 175, 174, 172, 171, 169, 168,
	166, 165, 163, 162, 160, 159, 157, 156, 154, 153, 151, 150, 146, 145, 148, 145,
	144, 123, 87, 124, 87, 125, 85, 126, 89, 127, 91, 128, 93, 129, 95, 130,
	97, 131, 99, 132, 101, 133, 103, 134, 105, 135, 107, 136, 109, 137, 111, 138,
	113, 139, 115, 140, 117, 141, 119, 142, 121, 143, 123, 143, 124, 29, 86, 29,
	122, 83, 120, 80, 118, 77, 116, 74, 114, 71, 112, 68, 110, 65, 108, 62,
	106, 59, 104, 56, 102, 53, 100, 50, 98, 47, 96, 44, 94, 41, 92, 38,
	90, 35, 88, 32, 84, 27, 86, 27, 29, 24, 28, 24, 82, 81, 79, 78,
	76, 75, 73, 72, 70, 69, 67, 66, 64, 63, 61, 60, 58, 57, 55, 54,
	52, 51, 49, 48, 46, 45, 43, 42, 40, 39, 37, 36, 34, 33, 31, 30,
	26, 25, 28, 25, 24
};

static QVector3D frustumVertices[8]; // 8 vertices of frustum
static QVector3D frustumNormals[6]; // 6 face normals of frustum


void drawCamera()
{


	glPushAttrib(GL_ALL_ATTRIB_BITS);

	int drawMode = 0;

	if (drawMode == 0)           // fill mode
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
	}
	else if (drawMode == 1)      // wireframe mode
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		//glDisable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);
	}
	else if (drawMode == 2)      // point mode
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
		//glDisable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);
	}





	float shininess = 32.0f;
	float diffuseColor[3] = { 1.0f, 1.0f, 1.0f };
	float specularColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };

	//// set specular and shiniess using glMaterial
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess); // range 0 ~ 128
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specularColor);

	// set ambient and diffuse color using glColorMaterial (gold-yellow)
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glColor3fv(diffuseColor);

	// start to render polygons
	glAssert(glEnableClientState(GL_NORMAL_ARRAY));
	glAssert(glEnableClientState(GL_VERTEX_ARRAY));

	glAssert(glNormalPointer(GL_FLOAT, 0, cameraNormals));
	glAssert(glVertexPointer(3, GL_FLOAT, 0, cameraVertices));

	glAssert(glDrawElements(GL_TRIANGLE_STRIP, 5, GL_UNSIGNED_INT, &cameraIndices[0]));
	glAssert(glDrawElements(GL_TRIANGLE_STRIP, 5, GL_UNSIGNED_INT, &cameraIndices[5]));
	glAssert(glDrawElements(GL_TRIANGLE_STRIP, 5, GL_UNSIGNED_INT, &cameraIndices[10]));
	glAssert(glDrawElements(GL_TRIANGLE_STRIP, 5, GL_UNSIGNED_INT, &cameraIndices[15]));
	glAssert(glDrawElements(GL_TRIANGLE_STRIP, 5, GL_UNSIGNED_INT, &cameraIndices[20]));
	glAssert(glDrawElements(GL_TRIANGLE_STRIP, 5, GL_UNSIGNED_INT, &cameraIndices[25]));
	glAssert(glDrawElements(GL_TRIANGLE_STRIP, 39, GL_UNSIGNED_INT, &cameraIndices[30]));
	glAssert(glDrawElements(GL_TRIANGLE_STRIP, 44, GL_UNSIGNED_INT, &cameraIndices[69]));
	glAssert(glDrawElements(GL_TRIANGLE_STRIP, 44, GL_UNSIGNED_INT, &cameraIndices[113]));
	glAssert(glDrawElements(GL_TRIANGLE_STRIP, 44, GL_UNSIGNED_INT, &cameraIndices[157]));
	glAssert(glDrawElements(GL_TRIANGLE_STRIP, 44, GL_UNSIGNED_INT, &cameraIndices[201]));

	glDisableClientState(GL_VERTEX_ARRAY);	// disable vertex arrays
	glDisableClientState(GL_NORMAL_ARRAY);	// disable normal arrays
	glPopAttrib();
}
///////////////////////////////////////////////////////////////////////////////
// compute 8 vertices of frustum
///////////////////////////////////////////////////////////////////////////////
void computeFrustumVertices(int projectionMode, float l, float r, float b, float t, float n, float f);

void computeFrustumVertices(int projectionMode, float l, float r, float b, float t, float n, float f)
{
	float ratio;
	float farLeft;
	float farRight;
	float farBottom;
	float farTop;

	// perspective mode
	if (projectionMode == 0)
		ratio = f / n;
	// orthographic mode
	else
		ratio = 1;
	farLeft = l * ratio;
	farRight = r * ratio;
	farBottom = b * ratio;
	farTop = t * ratio;

	// compute 8 vertices of the frustum
	// near top right
	frustumVertices[0][0] = r;
	frustumVertices[0][1] = t;
	frustumVertices[0][2] = -n;

	// near top left
	frustumVertices[1][0] = l;
	frustumVertices[1][1] = t;
	frustumVertices[1][2] = -n;

	// near bottom left
	frustumVertices[2][0] = l;
	frustumVertices[2][1] = b;
	frustumVertices[2][2] = -n;

	// near bottom right
	frustumVertices[3][0] = r;
	frustumVertices[3][1] = b;
	frustumVertices[3][2] = -n;

	// far top right
	frustumVertices[4][0] = farRight;
	frustumVertices[4][1] = farTop;
	frustumVertices[4][2] = -f;

	// far top left
	frustumVertices[5][0] = farLeft;
	frustumVertices[5][1] = farTop;
	frustumVertices[5][2] = -f;

	// far bottom left
	frustumVertices[6][0] = farLeft;
	frustumVertices[6][1] = farBottom;
	frustumVertices[6][2] = -f;

	// far bottom right
	frustumVertices[7][0] = farRight;
	frustumVertices[7][1] = farBottom;
	frustumVertices[7][2] = -f;

	// compute normals
	frustumNormals[0] = QVector3D::crossProduct((frustumVertices[5] - frustumVertices[1]), (frustumVertices[2] - frustumVertices[1]));
	frustumNormals[0].normalize();

	frustumNormals[1] = QVector3D::crossProduct((frustumVertices[3] - frustumVertices[0]), (frustumVertices[4] - frustumVertices[0]));
	frustumNormals[1].normalize();

	frustumNormals[2] = QVector3D::crossProduct((frustumVertices[6] - frustumVertices[2]), (frustumVertices[3] - frustumVertices[2]));
	frustumNormals[2].normalize();

	frustumNormals[3] = QVector3D::crossProduct((frustumVertices[4] - frustumVertices[0]), (frustumVertices[1] - frustumVertices[0]));
	frustumNormals[3].normalize();

	frustumNormals[4] = QVector3D::crossProduct((frustumVertices[1] - frustumVertices[0]), (frustumVertices[3] - frustumVertices[0]));
	frustumNormals[4].normalize();

	frustumNormals[5] = QVector3D::crossProduct((frustumVertices[7] - frustumVertices[4]), (frustumVertices[5] - frustumVertices[4]));
	frustumNormals[5].normalize();
}

void drawFrustum(int projectionMode, float l, float r, float b, float t, float n, float f)
{
	computeFrustumVertices(projectionMode, l, r, b, t, n, f);

	float colorLine1[4] = { 0.7f, 0.7f, 0.7f, 0.7f };
	float colorLine2[4] = { 0.2f, 0.2f, 0.2f, 0.7f };
	float colorPlane1[4] = { 0.5f, 0.5f, 0.5f, 0.5f };

	// draw lines
	glDisable(GL_LIGHTING);
	glDisable(GL_CULL_FACE);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// draw the edges around frustum
	if (projectionMode == 0)
	{
		glBegin(GL_LINES);
		glColor4fv(colorLine2);
		glVertex3f(0, 0, 0);
		glColor4fv(colorLine1);
		glVertex3fv(&frustumVertices[4][0]);

		glColor4fv(colorLine2);
		glVertex3f(0, 0, 0);
		glColor4fv(colorLine1);
		glVertex3fv(&frustumVertices[5][0]);

		glColor4fv(colorLine2);
		glVertex3f(0, 0, 0);
		glColor4fv(colorLine1);
		glVertex3fv(&frustumVertices[6][0]);

		glColor4fv(colorLine2);
		glVertex3f(0, 0, 0);
		glColor4fv(colorLine1);
		glVertex3fv(&frustumVertices[7][0]);
		glEnd();
	}
	else
	{
		glColor4fv(colorLine1);
		glBegin(GL_LINES);
		glVertex3fv(&frustumVertices[0][0]);
		glVertex3fv(&frustumVertices[4][0]);
		glVertex3fv(&frustumVertices[1][0]);
		glVertex3fv(&frustumVertices[5][0]);
		glVertex3fv(&frustumVertices[2][0]);
		glVertex3fv(&frustumVertices[6][0]);
		glVertex3fv(&frustumVertices[3][0]);
		glVertex3fv(&frustumVertices[7][0]);
		glEnd();
	}

	glColor4fv(colorLine1);
	glBegin(GL_LINE_LOOP);
	glVertex3fv(&frustumVertices[4][0]);
	glVertex3fv(&frustumVertices[5][0]);
	glVertex3fv(&frustumVertices[6][0]);
	glVertex3fv(&frustumVertices[7][0]);
	glEnd();

	glColor4fv(colorLine1);
	glBegin(GL_LINE_LOOP);
	glVertex3fv(&frustumVertices[0][0]);
	glVertex3fv(&frustumVertices[1][0]);
	glVertex3fv(&frustumVertices[2][0]);
	glVertex3fv(&frustumVertices[3][0]);
	glEnd();

	glEnable(GL_CULL_FACE);
	glEnable(GL_LIGHTING);

	// frustum is transparent.
	// Draw the frustum faces twice: backfaces first then frontfaces second.
	for (int i = 0; i < 2; ++i)
	{
		if (i == 0)
		{
			// for inside planes
			//glCullFace(GL_FRONT);
			//glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
		}
		else
		{
			// draw outside planes
			glCullFace(GL_BACK);
			glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
		}

		glColor4fv(colorPlane1);
		glBegin(GL_QUADS);
		// left
		glNormal3fv(&frustumNormals[0][0]);
		glVertex3fv(&frustumVertices[1][0]);
		glVertex3fv(&frustumVertices[5][0]);
		glVertex3fv(&frustumVertices[6][0]);
		glVertex3fv(&frustumVertices[2][0]);
		// right
		glNormal3fv(&frustumNormals[1][0]);
		glVertex3fv(&frustumVertices[0][0]);
		glVertex3fv(&frustumVertices[3][0]);
		glVertex3fv(&frustumVertices[7][0]);
		glVertex3fv(&frustumVertices[4][0]);
		// bottom
		glNormal3fv(&frustumNormals[2][0]);
		glVertex3fv(&frustumVertices[2][0]);
		glVertex3fv(&frustumVertices[6][0]);
		glVertex3fv(&frustumVertices[7][0]);
		glVertex3fv(&frustumVertices[3][0]);
		// top
		glNormal3fv(&frustumNormals[3][0]);
		glVertex3fv(&frustumVertices[0][0]);
		glVertex3fv(&frustumVertices[4][0]);
		glVertex3fv(&frustumVertices[5][0]);
		glVertex3fv(&frustumVertices[1][0]);
		// front
		glNormal3fv(&frustumNormals[4][0]);
		glVertex3fv(&frustumVertices[0][0]);
		glVertex3fv(&frustumVertices[1][0]);
		glVertex3fv(&frustumVertices[2][0]);
		glVertex3fv(&frustumVertices[3][0]);
		// back
		glNormal3fv(&frustumNormals[5][0]);
		glVertex3fv(&frustumVertices[7][0]);
		glVertex3fv(&frustumVertices[6][0]);
		glVertex3fv(&frustumVertices[5][0]);
		glVertex3fv(&frustumVertices[4][0]);
		glEnd();
	}
}
