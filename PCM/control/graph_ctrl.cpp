#include "graph_ctrl.hpp"

#include <iostream>

#include "animation/graph.hpp"
#include "global_datas/cuda_globals.hpp"
#include "global_datas/toolglobals.hpp"
#include "../meshes/mesh.hpp"
// -----------------------------------------------------------------------------
using namespace Tbx;
void Graph_ctrl::save_to_file(const char* filename) const
{
    g_graph->save_to_file(filename);
}

// -----------------------------------------------------------------------------

void Graph_ctrl::load_from_file(const char* filename) const{
    g_graph->load_from_file(filename);
    std::cout << "Loading from file: " << filename << std::endl;
    g_graph->set_offset_scale(g_mesh->get_offset(), g_mesh->get_scale());
}

// -----------------------------------------------------------------------------

bool Graph_ctrl::is_loaded(){
    return g_graph->nb_vertices() != 0;
}

// -----------------------------------------------------------------------------

int Graph_ctrl::push_vertex(const Vec3 &v)
{
    return g_graph->push_vertex(Vec3(v.x, v.y, v.z));
}

// -----------------------------------------------------------------------------

void Graph_ctrl::remove(int i)
{
    _selected_node = -1;
    return g_graph->remove_vertex(i);
}

// -----------------------------------------------------------------------------

int Graph_ctrl::push_edge(int v1, int v2)
{
    return g_graph->push_edge(Graph::Edge(v1, v2));
}

// -----------------------------------------------------------------------------

Vec3 Graph_ctrl::get_vertex(int id)
{
    Vec3 v = g_graph->get_vertex(id);
    return Vec3(v.x, v.y, v.z);
}

// -----------------------------------------------------------------------------

void Graph_ctrl::set_vertex(int id, Vec3 v){
    if( is_loaded() )
        g_graph->set_vertex( id, Vec3(v.x, v.y, v.z) );
}

// -----------------------------------------------------------------------------

void Graph_ctrl::center_vertex(int )
{
}

// -----------------------------------------------------------------------------

bool Graph_ctrl::select_node(int x, int y)
{
    float dst;
    //y = Cuda_ctrl::_display._height - y;
    int nearest = g_graph->get_window_nearest((float)x, (float)y, dst);
    if(dst < 8.f){
        _selected_node = nearest;
        return true;
    }
    return false;
}

// -----------------------------------------------------------------------------
