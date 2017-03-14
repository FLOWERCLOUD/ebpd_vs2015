#include "ray_marching.hpp"

#include "macros.hpp"
#include "cuda_ctrl.hpp"

#include "object.hpp"
#include "obj_implicit_tree.hpp"

#include "filters.hpp"
#include "toolbox/gl_utils/gltex2D.hpp"
#include  "raytracing_context.hpp"


// =============================================================================
namespace Raytracing {
// =============================================================================

Context g_raytracing_contex;

bool reset_buffs = true;

void reset_buffers(){ reset_buffs = true; }

Context& context(){ return g_raytracing_contex; }

// -----------------------------------------------------------------------------

/// Update buffers by raytracing the implicit primitive defined by the current
/// skeleton.
/// @param progressive : Activate the progressive raytracing Every call fills a
/// litlle bit more the buffers this is used for previsualisation. When the
/// scene parameters changes such as camera position resolution or the position
/// of the potential plane Buffers are erased and raytracing is done again
/// from the beginning
/// @return number of remaining calls to do to completely fill the buffers
int trace_nth_pass(const Obj_implicit_tree* tree,
                   float4* d_rendu,
                   float* d_depth,
                   int width,
                   int height,
                   bool progressive,
                   bool reset)
{
    Context& ctx = context();

    ctx.pbo = PBO_data(d_rendu, d_depth, 0, 0, width * MULTISAMPX, height * MULTISAMPY);

    int nb_samples = g_raytracing_contex.max_res_nth_pass;
    static int passes = 0;

    nb_samples = nb_samples>width  ? width  : nb_samples;
    nb_samples = nb_samples>height ? height : nb_samples;

    int3 steps = {width/nb_samples, height/nb_samples, passes};

    dim3 dimBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);
    dim3 dimGrid((width  / steps.x * MULTISAMPX + dimBlock.x-1) / dimBlock.x,
                 (height / steps.y * MULTISAMPY + dimBlock.y-1) / dimBlock.y);

    // If any change in the scene occured we start raytracing again from the
    // begining
    if( reset )
    {
        clean_buffers(d_rendu, d_depth, ctx.cam._far, width, height);
        passes  = 0;
        steps.z = 0;
    }

    int nb_pass_max = (steps.x * steps.y);

    // if there are still empty pixels we raytrace them
    while( passes < nb_pass_max )
    {
        ctx.grid  = dimGrid;
        ctx.block = dimBlock;
        ctx.steps = steps;

        tree->trace( ctx );

        passes++;
        steps.z = passes;

        // When enabled we do the bloom effect at the last pass
        // FIXME: blooms needs the img_buff and bloom_buff from the current rendering context
//        if(_display._bloom && passes == nb_pass_max)
//            do_bloom(d_rendu, width*MULTISAMPX, height*MULTISAMPY);

        // progressive mode we do not raytrace all the pixels at the same time
        if(progressive) break;
    }

    return nb_pass_max - passes;
}

// -----------------------------------------------------------------------------

/// Flaten and smooth if necessary the buffer.
void filter_buffers(float4* d_in_color,
                    float*  d_in_depth,
                    int*      d_out_rgb24,
                    unsigned* d_out_depth,
                    int width,
                    int height,
                    int passes,
                    bool dither = false)
{
    dim3 dimBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);
    dim3 dimGrid((width * MULTISAMPX + dimBlock.x-1) / dimBlock.x,
                  (height* MULTISAMPY + dimBlock.y-1) / dimBlock.y);

    bool do_filter  = passes > 0; // Finish we do not filter
    int block_x   = width  / Cuda_ctrl::_display._nb_samples_res;
    int block_y   = height / Cuda_ctrl::_display._nb_samples_res;
    // The filter size is proportionnal to the square root of
    // number of passes remaining
    float percent = sqrt((float)max(passes, 1) / (float)(block_y*block_x));
    // Filter size is maximum 16 because higher sizes are really too slow
    int filter_size = min(16, 1+max( (int)(block_x*percent), (int)(block_y*percent)));

    flatten_image<<< dimGrid, dimBlock >>>
                                        (d_in_color,
                                          d_in_depth,
                                          d_out_rgb24,
                                          d_out_depth,
                                          width*MULTISAMPX,
                                          height*MULTISAMPY,
                                          do_filter,
                                          filter_size,
                                          dither);
}

// -----------------------------------------------------------------------------

bool trace_tree(const Obj_implicit_tree* tree)
{
    Render_context* ctx = context().render_ctx;

    const int width  = ctx->width();
    const int height = ctx->height();

    bool refresh = false;
    int*      pbo_color = 0;
    unsigned* pbo_depth = 0;
    ctx->pbo_color()->cuda_map_to( pbo_color );
    ctx->pbo_depth()->cuda_map_to( pbo_depth );
    if( ctx->_raytrace && tree->state(Obj::RENDER) )
    {
        bool prog  = g_raytracing_contex.progressive;
        bool reset = /* has_changed(cam, ctx.plane_n, ctx.plane_org, width, height) || */ reset_buffs;
        reset_buffs = false;

        int passes = !trace_nth_pass(tree,ctx->d_render_buff(), ctx->d_depth_buff(), width, height, prog, reset);

        filter_buffers(ctx->d_render_buff(), ctx->d_depth_buff(), pbo_color, pbo_depth, width, height, passes);

        refresh = passes != 0;

        ctx->pbo_color()->cuda_unmap();
        ctx->pbo_depth()->cuda_unmap();

        ctx->pbo_color()->bind();
        ctx->frame_tex()->bind();
        ctx->frame_tex()->allocate(GL_UNSIGNED_BYTE, GL_RGBA);
        ctx->pbo_color()->unbind();

        EnableProgram();
        draw_quad();
        DisableProgram();
    }
    else
    {
        Color cl = context().background;
        float4 cl_color = {cl.r, cl.g, cl.b, cl.a};
        clean_pbos(pbo_color, pbo_depth, width, height, cl_color);
        ctx->pbo_color()->cuda_unmap();
        ctx->pbo_depth()->cuda_unmap();
    }
    return refresh;
}

// -----------------------------------------------------------------------------

void clear_buffers()
{
    Render_context* ctx = context().render_ctx;
    const int width  = ctx->width();
    const int height = ctx->height();

    int*      pbo_color = 0;
    unsigned* pbo_depth = 0;
    ctx->pbo_color()->cuda_map_to( pbo_color );
    ctx->pbo_depth()->cuda_map_to( pbo_depth );

    Color cl = context().background;
    float4 cl_color = {cl.r, cl.g, cl.b, cl.a};
    clean_pbos(pbo_color, pbo_depth, width, height, cl_color);
    ctx->pbo_color()->cuda_unmap();
    ctx->pbo_depth()->cuda_unmap();
}

}// END RAYTRACING =============================================================
