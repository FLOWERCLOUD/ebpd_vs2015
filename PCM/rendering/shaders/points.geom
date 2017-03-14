#version 120
#extension GL_EXT_geometry_shader4 : enable
#extension GL_EXT_gpu_shader4 : enable

//varying in vec3 vnormal[];
//varying in vec4 vposition[];

//varying out vec3 normal;
//varying out vec4 position;

float point_size = 0.2f;

void main()
{
    gl_TexCoord[0] = vec4(0., 1., 0., 1.);
    //normal = vnormal[0];
    gl_Position = gl_PositionIn[0] + vec4(-point_size, -point_size, 0., 0.);
    EmitVertex();
    gl_TexCoord[0] = vec4(0., 0., 0., 1.);
    //normal = vnormal[0];
    gl_Position = gl_PositionIn[0] + vec4(-point_size,  point_size, 0., 0.);
    EmitVertex();
    gl_TexCoord[0] = vec4(1., 1., 0., 1.);
    //normal = vnormal[0];
    gl_Position = gl_PositionIn[0] + vec4( point_size, -point_size, 0., 0.);
    EmitVertex();
    gl_TexCoord[0] = vec4(1., 0., 0., 1.);
    //normal = vnormal[0];
    gl_Position = gl_PositionIn[0] + vec4( point_size,  point_size, 0., 0.);
    EmitVertex();
    EndPrimitive();
}
