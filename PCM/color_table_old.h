#ifndef _COLOR_TABLE_H
#define _COLOR_TABLE_H
#include "basic_types.h"

#define SELECTED_COLOR ColorType(1.0f, 0.0f, 0.0f, 1.0f)
#define HIGHTLIGHTED_COLOR ColorType(0.0f, 0.0f, 0.0f, 1.0f)


namespace Color_Utility
{
	// 256 colors
	static const ScalarType color_table[256][3] = {
		0.0f, 0.0f, 0.5156f
		, 0.0f, 0.0f, 0.5313f
		, 0.0f, 0.0f, 0.5469f
		, 0.0f, 0.0f, 0.5625f
		, 0.0f, 0.0f, 0.5781f
		, 0.0f, 0.0f, 0.5938f
		, 0.0f, 0.0f, 0.6094f
		, 0.0f, 0.0f, 0.6250f
		, 0.0f, 0.0f, 0.6406f
		, 0.0f, 0.0f, 0.6563f
		, 0.0f, 0.0f, 0.6719f
		, 0.0f, 0.0f, 0.6875f
		, 0.0f, 0.0f, 0.7031f
		, 0.0f, 0.0f, 0.7188f
		, 0.0f, 0.0f, 0.7344f
		, 0.0f, 0.0f, 0.7500f
		, 0.0f, 0.0f, 0.7656f
		, 0.0f, 0.0f, 0.7813f
		, 0.0f, 0.0f, 0.7969f
		, 0.0f, 0.0f, 0.8125f
		, 0.0f, 0.0f, 0.8281f
		, 0.0f, 0.0f, 0.8438f
		, 0.0f, 0.0f, 0.8594f
		, 0.0f, 0.0f, 0.8750f
		, 0.0f, 0.0f, 0.8906f
		, 0.0f, 0.0f, 0.9063f
		, 0.0f, 0.0f, 0.9219f
		, 0.0f, 0.0f, 0.9375f
		, 0.0f, 0.0f, 0.9531f
		, 0.0f, 0.0f, 0.9688f
		, 0.0f, 0.0f, 0.9844f
		, 0.0f, 0.0f, 1.0000f
		, 0.0f, 0.0156f, 1.0f
		, 0.0f, 0.0313f, 1.0f
		, 0.0f, 0.0469f, 1.0f
		, 0.0f, 0.0625f, 1.0f
		, 0.0f, 0.0781f, 1.0f
		, 0.0f, 0.0938f, 1.0f
		, 0.0f, 0.1094f, 1.0f
		, 0.0f, 0.1250f, 1.0f
		, 0.0f, 0.1406f, 1.0f
		, 0.0f, 0.1563f, 1.0f
		, 0.0f, 0.1719f, 1.0f
		, 0.0f, 0.1875f, 1.0f
		, 0.0f, 0.2031f, 1.0f
		, 0.0f, 0.2188f, 1.0f
		, 0.0f, 0.2344f, 1.0f
		, 0.0f, 0.2500f, 1.0f
		, 0.0f, 0.2656f, 1.0f
		, 0.0f, 0.2813f, 1.0f
		, 0.0f, 0.2969f, 1.0f
		, 0.0f, 0.3125f, 1.0f
		, 0.0f, 0.3281f, 1.0f
		, 0.0f, 0.3438f, 1.0f
		, 0.0f, 0.3594f, 1.0f
		, 0.0f, 0.3750f, 1.0f
		, 0.0f, 0.3906f, 1.0f
		, 0.0f, 0.4063f, 1.0f
		, 0.0f, 0.4219f, 1.0f
		, 0.0f, 0.4375f, 1.0f
		, 0.0f, 0.4531f, 1.0f
		, 0.0f, 0.4688f, 1.0f
		, 0.0f, 0.4844f, 1.0f
		, 0.0f, 0.5000f, 1.0f
		, 0.0f, 0.5156f, 1.0f
		, 0.0f, 0.5313f, 1.0f
		, 0.0f, 0.5469f, 1.0f
		, 0.0f, 0.5625f, 1.0f
		, 0.0f, 0.5781f, 1.0f
		, 0.0f, 0.5938f, 1.0f
		, 0.0f, 0.6094f, 1.0f
		, 0.0f, 0.6250f, 1.0f
		, 0.0f, 0.6406f, 1.0f
		, 0.0f, 0.6563f, 1.0f
		, 0.0f, 0.6719f, 1.0f
		, 0.0f, 0.6875f, 1.0f
		, 0.0f, 0.7031f, 1.0f
		, 0.0f, 0.7188f, 1.0f
		, 0.0f, 0.7344f, 1.0f
		, 0.0f, 0.7500f, 1.0f
		, 0.0f, 0.7656f, 1.0f
		, 0.0f, 0.7813f, 1.0f
		, 0.0f, 0.7969f, 1.0f
		, 0.0f, 0.8125f, 1.0f
		, 0.0f, 0.8281f, 1.0f
		, 0.0f, 0.8438f, 1.0f
		, 0.0f, 0.8594f, 1.0f
		, 0.0f, 0.8750f, 1.0f
		, 0.0f, 0.8906f, 1.0f
		, 0.0f, 0.9063f, 1.0f
		, 0.0f, 0.9219f, 1.0f
		, 0.0f, 0.9375f, 1.0f
		, 0.0f, 0.9531f, 1.0f
		, 0.0f, 0.9688f, 1.0f
		, 0.0f, 0.9844f, 1.0f
		, 0.0f, 1.0000f, 1.0f
		, 0.0156f, 1.0000f, 0.9844f
		, 0.0313f, 1.0000f, 0.9688f
		, 0.0469f, 1.0000f, 0.9531f
		, 0.0625f, 1.0000f, 0.9375f
		, 0.0781f, 1.0000f, 0.9219f
		, 0.0938f, 1.0000f, 0.9063f
		, 0.1094f, 1.0000f, 0.8906f
		, 0.1250f, 1.0000f, 0.8750f
		, 0.1406f, 1.0000f, 0.8594f
		, 0.1563f, 1.0000f, 0.8438f
		, 0.1719f, 1.0000f, 0.8281f
		, 0.1875f, 1.0000f, 0.8125f
		, 0.2031f, 1.0000f, 0.7969f
		, 0.2188f, 1.0000f, 0.7813f
		, 0.2344f, 1.0000f, 0.7656f
		, 0.2500f, 1.0000f, 0.7500f
		, 0.2656f, 1.0000f, 0.7344f
		, 0.2813f, 1.0000f, 0.7188f
		, 0.2969f, 1.0000f, 0.7031f
		, 0.3125f, 1.0000f, 0.6875f
		, 0.3281f, 1.0000f, 0.6719f
		, 0.3438f, 1.0000f, 0.6563f
		, 0.3594f, 1.0000f, 0.6406f
		, 0.3750f, 1.0000f, 0.6250f
		, 0.3906f, 1.0000f, 0.6094f
		, 0.4063f, 1.0000f, 0.5938f
		, 0.4219f, 1.0000f, 0.5781f
		, 0.4375f, 1.0000f, 0.5625f
		, 0.4531f, 1.0000f, 0.5469f
		, 0.4688f, 1.0000f, 0.5313f
		, 0.4844f, 1.0000f, 0.5156f
		, 0.5000f, 1.0000f, 0.5000f
		, 0.5156f, 1.0000f, 0.4844f
		, 0.5313f, 1.0000f, 0.4688f
		, 0.5469f, 1.0000f, 0.4531f
		, 0.5625f, 1.0000f, 0.4375f
		, 0.5781f, 1.0000f, 0.4219f
		, 0.5938f, 1.0000f, 0.4063f
		, 0.6094f, 1.0000f, 0.3906f
		, 0.6250f, 1.0000f, 0.3750f
		, 0.6406f, 1.0000f, 0.3594f
		, 0.6563f, 1.0000f, 0.3438f
		, 0.6719f, 1.0000f, 0.3281f
		, 0.6875f, 1.0000f, 0.3125f
		, 0.7031f, 1.0000f, 0.2969f
		, 0.7188f, 1.0000f, 0.2813f
		, 0.7344f, 1.0000f, 0.2656f
		, 0.7500f, 1.0000f, 0.2500f
		, 0.7656f, 1.0000f, 0.2344f
		, 0.7813f, 1.0000f, 0.2188f
		, 0.7969f, 1.0000f, 0.2031f
		, 0.8125f, 1.0000f, 0.1875f
		, 0.8281f, 1.0000f, 0.1719f
		, 0.8438f, 1.0000f, 0.1563f
		, 0.8594f, 1.0000f, 0.1406f
		, 0.8750f, 1.0000f, 0.1250f
		, 0.8906f, 1.0000f, 0.1094f
		, 0.9063f, 1.0000f, 0.0938f
		, 0.9219f, 1.0000f, 0.0781f
		, 0.9375f, 1.0000f, 0.0625f
		, 0.9531f, 1.0000f, 0.0469f
		, 0.9688f, 1.0000f, 0.0313f
		, 0.9844f, 1.0000f, 0.0156f
		, 1.0000f, 1.0000f, 0.0f
		, 1.0000f, 0.9844f, 0.0f
		, 1.0000f, 0.9688f, 0.0f
		, 1.0000f, 0.9531f, 0.0f
		, 1.0000f, 0.9375f, 0.0f
		, 1.0000f, 0.9219f, 0.0f
		, 1.0000f, 0.9063f, 0.0f
		, 1.0000f, 0.8906f, 0.0f
		, 1.0000f, 0.8750f, 0.0f
		, 1.0000f, 0.8594f, 0.0f
		, 1.0000f, 0.8438f, 0.0f
		, 1.0000f, 0.8281f, 0.0f
		, 1.0000f, 0.8125f, 0.0f
		, 1.0000f, 0.7969f, 0.0f
		, 1.0000f, 0.7813f, 0.0f
		, 1.0000f, 0.7656f, 0.0f
		, 1.0000f, 0.7500f, 0.0f
		, 1.0000f, 0.7344f, 0.0f
		, 1.0000f, 0.7188f, 0.0f
		, 1.0000f, 0.7031f, 0.0f
		, 1.0000f, 0.6875f, 0.0f
		, 1.0000f, 0.6719f, 0.0f
		, 1.0000f, 0.6563f, 0.0f
		, 1.0000f, 0.6406f, 0.0f
		, 1.0000f, 0.6250f, 0.0f
		, 1.0000f, 0.6094f, 0.0f
		, 1.0000f, 0.5938f, 0.0f
		, 1.0000f, 0.5781f, 0.0f
		, 1.0000f, 0.5625f, 0.0f
		, 1.0000f, 0.5469f, 0.0f
		, 1.0000f, 0.5313f, 0.0f
		, 1.0000f, 0.5156f, 0.0f
		, 1.0000f, 0.5000f, 0.0f
		, 1.0000f, 0.4844f, 0.0f
		, 1.0000f, 0.4688f, 0.0f
		, 1.0000f, 0.4531f, 0.0f
		, 1.0000f, 0.4375f, 0.0f
		, 1.0000f, 0.4219f, 0.0f
		, 1.0000f, 0.4063f, 0.0f
		, 1.0000f, 0.3906f, 0.0f
		, 1.0000f, 0.3750f, 0.0f
		, 1.0000f, 0.3594f, 0.0f
		, 1.0000f, 0.3438f, 0.0f
		, 1.0000f, 0.3281f, 0.0f
		, 1.0000f, 0.3125f, 0.0f
		, 1.0000f, 0.2969f, 0.0f
		, 1.0000f, 0.2813f, 0.0f
		, 1.0000f, 0.2656f, 0.0f
		, 1.0000f, 0.2500f, 0.0f
		, 1.0000f, 0.2344f, 0.0f
		, 1.0000f, 0.2188f, 0.0f
		, 1.0000f, 0.2031f, 0.0f
		, 1.0000f, 0.1875f, 0.0f
		, 1.0000f, 0.1719f, 0.0f
		, 1.0000f, 0.1563f, 0.0f
		, 1.0000f, 0.1406f, 0.0f
		, 1.0000f, 0.1250f, 0.0f
		, 1.0000f, 0.1094f, 0.0f
		, 1.0000f, 0.0938f, 0.0f
		, 1.0000f, 0.0781f, 0.0f
		, 1.0000f, 0.0625f, 0.0f
		, 1.0000f, 0.0469f, 0.0f
		, 1.0000f, 0.0313f, 0.0f
		, 1.0000f, 0.0156f, 0.0f
		, 1.0000f, 0.0f, 0.0f
		, 0.9844f, 0.0f, 0.0f
		, 0.9688f, 0.0f, 0.0f
		, 0.9531f, 0.0f, 0.0f
		, 0.9375f, 0.0f, 0.0f
		, 0.9219f, 0.0f, 0.0f
		, 0.9063f, 0.0f, 0.0f
		, 0.8906f, 0.0f, 0.0f
		, 0.8750f, 0.0f, 0.0f
		, 0.8594f, 0.0f, 0.0f
		, 0.8438f, 0.0f, 0.0f
		, 0.8281f, 0.0f, 0.0f
		, 0.8125f, 0.0f, 0.0f
		, 0.7969f, 0.0f, 0.0f
		, 0.7813f, 0.0f, 0.0f
		, 0.7656f, 0.0f, 0.0f
		, 0.7500f, 0.0f, 0.0f
		, 0.7344f, 0.0f, 0.0f
		, 0.7188f, 0.0f, 0.0f
		, 0.7031f, 0.0f, 0.0f
		, 0.6875f, 0.0f, 0.0f
		, 0.6719f, 0.0f, 0.0f
		, 0.6563f, 0.0f, 0.0f
		, 0.6406f, 0.0f, 0.0f
		, 0.6250f, 0.0f, 0.0f
		, 0.6094f, 0.0f, 0.0f
		, 0.5938f, 0.0f, 0.0f
		, 0.5781f, 0.0f, 0.0f
		, 0.5625f, 0.0f, 0.0f
		, 0.5469f, 0.0f, 0.0f
		, 0.5313f, 0.0f, 0.0f
		, 0.5156f, 0.0f, 0.0f
		, 0.5000f, 0.0f, 0.0f
	};

	ColorType		color_from_table(IndexType index) ;
	ColorType		random_color_from_table();
	ColorType		span_color_from_table(IndexType);
}

#endif