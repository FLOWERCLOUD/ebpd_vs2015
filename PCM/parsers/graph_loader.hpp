#ifndef GRAPH_LOADER_HPP__
#define GRAPH_LOADER_HPP__

#include "loader.hpp"
//#include "graph.hpp"

// =============================================================================
namespace Loader {
// =============================================================================

class Graph_file : public Base_loader {
public:
    Graph_file(const std::string& file_name) : Base_loader( file_name )
    { import_file(file_name);  }

    /// The loader type
    Loader_t type() const { return SKEL; }

    bool import_file(const std::string& file_path){
        Base_loader::update_paths( file_path );
//        _graph.clear();
//        _graph.load_from_file( file_path.c_str() );
        return true;
    }

    bool export_file(const std::string& file_path){
        Base_loader::update_paths( file_path );
//        _graph.save_to_file( file_path.c_str() );
        return true;
    }

    ///// Fill the scene tree with the
    //EObj::Flags fill_scene(Scene_tree& tree, EObj::Flags flag = 0){

    //    if( EObj::test(flag, EObj::SKELETON) || flag == 0)
    //    {
    //        Obj_skeleton* skel = new Obj_skeleton();
    //        skel->load( _graph );
    //        tree.register_obj( skel );
    //        return EObj::SKELETON;
    //    }

    //    return EObj::NONE;
    //}

    /// @return parsed animations or NULL.
    void get_anims(std::vector<Base_anim_eval*>& anims) const { anims.clear(); }

    /// transform internal representation into generic representation
    /// which are the same here.
//    void get_graph(Graph& graph) const { graph = _graph; }

private:
//    Graph _graph;
};

}// END Loader =================================================================

#endif // GRAPH_LOADER_HPP__
