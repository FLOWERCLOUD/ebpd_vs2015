#include "parsers/weights_loader.hpp"

#include <cstring>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cassert>

// =============================================================================
namespace Loader {
// =============================================================================

// =============================================================================
namespace Weights_loader {
// =============================================================================

void load(const char* filename, int nb_vert, std::vector< std::map<int, float> >& weights)
{
    using namespace std;

    ifstream file(filename);

    weights.reserve( nb_vert );

    if(!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        assert(false);
    }

    int len = strlen(filename);
    bool file_has_commas = (filename[len-4] == '.') & (filename[len-3] == 'c') & (filename[len-2] == 's') & (filename[len-1] == 'v');

    for(int i = 0; i < nb_vert; i++)
    {
        std::string str_line;
        std::getline(file, str_line);
        std::stringbuf current_line_sb(str_line, ios_base::in);

        istream current_line(&current_line_sb);
        int j = 0;
        float weight, sum_weights = 0.f;

        weights.push_back( std::map<int, float>() );

        while(!current_line.eof() && !str_line.empty())
        {
            current_line >> j;

            if(file_has_commas) current_line.ignore(1,',');

            current_line >> weight;

            if(file_has_commas) current_line.ignore(1,',');

            current_line.ignore(10,' ');

            if(current_line.peek() == '\r') current_line.ignore(1,'\r');

            if(current_line.peek() == '\n') current_line.ignore(1,'\n');

            weights[i][j] = weight;
            sum_weights += weight;
        }

        if((sum_weights > 1.0001f) || (sum_weights < -0.0001f)){
            std::cerr << "WARNING: imported ssd weights does not sum to one ";
            std::cerr << "(line " << (i+1) << ")" << std::endl;
        }
    } // END FOR NB lINES

    cout << "file \"" << filename << "\" loaded successfully" << endl;
    file.close();
}

}// END Weights_loader =========================================================

}// END LOADER NAMESPACE =======================================================
