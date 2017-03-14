#ifndef _CACULATE_GRODISTANCE
#define  _CACULATE_GRODISTANCE
#include <vector>
void caculteGeodistance(
	const std::vector<double>& points,
	const std::vector<unsigned>& faces,
	int targetVertex,
	std::vector<float>& distance,
	int propagate_depth,
	void* drawbackback =NULL ,int drawback_index = -1);



#endif // !_CACULATE_GRODISTANCE
