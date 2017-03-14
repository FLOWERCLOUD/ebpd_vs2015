#ifndef AUTO_RIG_HPP__
#define AUTO_RIG_HPP__

#include <vector>
#include "toolbox/maths/vec3.hpp"

/**
  Automatic SSD weight computation from :
  @code
  @inproceedings{Baran:2007:ARA:1275808.1276467,
    author = {Baran, Ilya and Popovi\'{c}, Jovan},
    title = {Automatic rigging and animation of 3D characters},
    booktitle = {ACM SIGGRAPH 2007 papers},
    series = {SIGGRAPH '07},
    year = {2007},
    location = {San Diego, California},
    articleno = {72},
    url = {http://doi.acm.org/10.1145/1275808.1276467},
    doi = {http://doi.acm.org/10.1145/1275808.1276467},
    acmid = {1276467},
    publisher = {ACM},
    address = {New York, NY, USA},
    keywords = {animation, deformations, geometric modeling},
  }
  @endcode



*/
void rig(const std::vector< Vec3             >& vertices,
         const std::vector< std::vector<int>    >& edges,
         const std::vector< std::vector<double> >& boneDists,
         const std::vector< std::vector<bool>   >& boneVis,
         std::vector<std::vector<std::pair<int, double> > >& nzweights,
         double heat);


#endif // AUTO_RIG_HPP__
