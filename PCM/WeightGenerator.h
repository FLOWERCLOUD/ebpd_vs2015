#pragma once
#include <vector>
#include <string>
class Sample;
class Handle;
void computeWeights(Sample* sample, std::vector<Handle*>& _handles, std::vector<std::vector<float>>& _wieghts);
void writeWeightToFile(std::string& path, std::vector<std::vector<float>>& _weights);


