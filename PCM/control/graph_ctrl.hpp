#ifndef GRAPH_CTRL_HPP_
#define GRAPH_CTRL_HPP_

/** @brief graph controller
    This class is an utility to control the graph.


*/

#include "toolbox/maths/vec3.hpp"

class Graph_ctrl {
public:

    Graph_ctrl() :
        _selected_node(-1)
    {
    }

    void save_to_file(const char* filename) const;
    void load_from_file(const char* filename) const;

    bool is_loaded();

    /// Push a vertex in the vertices list
    int push_vertex(const Tbx::Vec3& v);

    void remove(int i);

    /// Push an edge in the edges list
    int push_edge(int v1, int v2);

   Tbx:: Vec3 get_vertex(int id);

    void set_vertex(int id, Tbx::Vec3 v);

    /// Try to center the node inside the mesh
    void center_vertex(int id);

    int  get_selected_node()     { return _selected_node; }
    void set_selected_node(int n){ _selected_node = n;    }
    void reset_selection()       { _selected_node = -1;   }

    /// @return if a vertex has been selected
    bool select_node(int x, int y);

private:
    int _selected_node;
};



#endif // GRAPH_CTRL_HPP_
