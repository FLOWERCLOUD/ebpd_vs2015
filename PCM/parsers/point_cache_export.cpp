#include "parsers/point_cache_export.hpp"

#include <iostream>
#include <fstream>
#include <cassert>
#include <algorithm>

#include "endianess.hpp"
#include "toolbox/maths/vec3.hpp"////////

// =============================================================================
namespace Loader {
// =============================================================================
using namespace Tbx;
Point_cache_file::Point_cache_file(int nb_points, int nb_frame_hint)
{
    assert(nb_points > 0);
    if(nb_frame_hint > 0)
    {
        _nb_points = nb_points;
        _frames.reserve(nb_frame_hint);
//        for(int i = 0; i < nb_frame_hint; i++)
//            _frames[i].resize(nb_points);
    }
}

// -----------------------------------------------------------------------------

void Point_cache_file::add_frame(float* points, int offset, int stride)
{
    _frames.push_back(std::vector<float>());
    const int frame_idx = _frames.size()-1;
    _frames[frame_idx].resize(_nb_points*3);
    for(int i = offset, j = 0; i < (_nb_points + offset); i++, j++)
    {
        _frames[frame_idx][j*3  ] = points[i*(3+stride)  ];
        _frames[frame_idx][j*3+1] = points[i*(3+stride)+1];
        _frames[frame_idx][j*3+2] = points[i*(3+stride)+2];
    }
}

// -----------------------------------------------------------------------------

/*
MDD file format

The first thing to note is that the MDD file format is Motorola Big Endian byte
order as opposed to the Intel Little Endian standard.
So however you implement the structure below, you must come up with an algorithm
to flip the bytes during file IO.

The data structure is like so:
typedef Struct{
    int totalframes;
    int totalPoints;
    float *Times; //time for each frame
    float **points[3];
}mddstruct;


and the data is written like so:


totalframes
totalPoints
Times
while(!totalframes)
{
    while(!totalPoints)
    {
        write point[frame][point][axis];
        point++;
    }
    frame++;
}
*/

// -----------------------------------------------------------------------------

float filter(int frame)
{
    float x = (float)frame;
    return exp( - (x*x)/100.f );
}

// -----------------------------------------------------------------------------

static Vec3 to_point(int p, const std::vector<float>& points )
{
    return Vec3( points[p*3], points[p*3 + 1], points[p*3 + 2] );
}

// -----------------------------------------------------------------------------

static void denoise_frames(const std::vector< std::vector<float> >& frames,
                           std::vector< std::vector<float> >& smooth_frames)
{
    for(int f = 1; f < ((int)frames.size() - 1); f++)
    {
        std::cout << "frame:" << f << std::endl;

        for(int p = 0; p < (int)frames[f].size()/3; p++)
        {
            Vec3 avg_pos = to_point(p, frames[f]);
            int acc = 1;
            float sum = filter(0);
            float val = 0.f;

            // ignore central point
#if 1
            avg_pos = Vec3::zero();
            sum = 0.f;
#endif

            while( (val = filter(acc)) > 0.01f || (acc >  (int)frames[f].size()) )
            {
                Vec3 pos1 = to_point(p, frames[ std::min(f + acc, (int)frames.size() - 1) ]);
                Vec3 pos2 = to_point(p, frames[ std::max(f - acc, 0)                      ]);

                avg_pos += pos1 * val;
                avg_pos += pos2 * val;

                sum += val * 2.f;

                acc++;
            }

            avg_pos /= sum;

            smooth_frames[f][p*3 + 0] = avg_pos.x;
            smooth_frames[f][p*3 + 1] = avg_pos.y;
            smooth_frames[f][p*3 + 2] = avg_pos.z;
        }
    }

}

void Point_cache_file::export_mdd(const std::string& path_name)
{
    {
        std::ofstream file(path_name.c_str(), std::ios::binary|std::ios::trunc);

        if(!file.is_open())
        {
            std::cout << "ERROR: can't create/open " << path_name << std::endl;
            return;
        }

        int ibuff = Endianess::big_long( (int)_frames.size() );
        file.write((char*)&ibuff, 4);
        ibuff = Endianess::big_long( (int)(_nb_points)  );
        file.write((char*)&ibuff, 4);

        float fbuff[3] = {Endianess::big_float(0.1f), 0.f, 0.f};
        for(unsigned f = 0; f < _frames.size(); f++)
            file.write((char*)fbuff, 4);

        for(unsigned f = 0; f < _frames.size(); f++)
        {
            for(int p = 0; p < _nb_points; p++)
            {
                fbuff[0] = Endianess::big_float( _frames[f][p*3    ] );
                fbuff[1] = Endianess::big_float( _frames[f][p*3 + 1] );
                fbuff[2] = Endianess::big_float( _frames[f][p*3 + 2] );
                file.write((char*)fbuff, 4*3);
            }
        }
        file.close();
    }


    // HACK write smooth out frames: ////////////////////////////////////////////////
    {
        std::vector< std::vector<float> > smooth_frames( _frames.size() );
        for( int i = 0; i < (int)_frames.size(); i++ )
            smooth_frames[i] = _frames[i];

        std::cout << "pipo" << std::endl;
        denoise_frames( _frames, smooth_frames );
        std::cout << "popo" << std::endl;

        std::cout << path_name << std::endl;

        std::string new_path = path_name.substr(0, path_name.size()-4)+"_smooth.mdd";

        std::cout << new_path << std::endl;


        std::ofstream file( (new_path).c_str(), std::ios::binary|std::ios::trunc);

        if( !file.is_open() )
        {
            std::cout << "ERROR: can't create/open " << path_name << std::endl;
            return;
        }

        int ibuff = Endianess::big_long( (int)smooth_frames.size() );
        file.write((char*)&ibuff, 4);
        ibuff = Endianess::big_long( (int)(_nb_points)  );
        file.write((char*)&ibuff, 4);

        float fbuff[3] = {Endianess::big_float(0.1f), 0.f, 0.f};
        for(unsigned f = 0; f < smooth_frames.size(); f++)
            file.write((char*)fbuff, 4);

        for(unsigned f = 0; f < smooth_frames.size(); f++)
        {
            for(int p = 0; p < _nb_points; p++)
            {
                fbuff[0] = Endianess::big_float( smooth_frames[f][p*3    ] );
                fbuff[1] = Endianess::big_float( smooth_frames[f][p*3 + 1] );
                fbuff[2] = Endianess::big_float( smooth_frames[f][p*3 + 2] );
                file.write((char*)fbuff, 4*3);
            }
        }
        file.close();
    }

}

}// END Loader =================================================================
