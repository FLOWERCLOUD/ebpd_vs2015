#version 330

// attributes
layout(location = 0) in vec3 i_position; // xyz - position
layout(location = 1) in vec3 i_normal; // xyz - normal
layout(location = 2) in vec3 i_tangent; // xyz - tangent, w - handedness
layout(location = 3) in vec3 i_bi_tangent;

// matrices
uniform mat4 u_model_mat;
//uniform mat3 u_normal_mat;  //not use

// data to geometry shader
out vec3	o_normal;
out vec3 o_tangent;
out vec3 o_bitangent;

///////////////////////////////////////////////////////////////////

void main(void)
{
   // position to world coordinates
   gl_Position = u_model_mat * vec4(i_position, 1.0);
   vec4 i_tangent = vec4(i_tangent,1.0f);

   // normal, tangent and bitangent in world coordinates
   mat3 tmp_normal_mat = mat3(transpose(inverse(u_model_mat)));
   o_normal	= normalize(tmp_normal_mat * i_normal);
 //  o_normal	= normalize(u_normal_mat * i_normal);
 //  o_tangent	= normalize(u_normal_mat * i_tangent.xyz);
 //  o_bitangent	= cross(o_normal, o_tangent) * i_tangent.w;
   o_tangent	= normalize(tmp_normal_mat * i_tangent.xyz);
//   o_bitangent	= normalize(cross(o_normal, o_tangent));
	o_bitangent = normalize(tmp_normal_mat *i_bi_tangent);
}