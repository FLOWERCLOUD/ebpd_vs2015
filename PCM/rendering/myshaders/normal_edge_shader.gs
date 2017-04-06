#version 330

// type of input primitives
layout(triangles) in;

// type of output primitives and maximum number of new vertices
// each line requires two vertices
// vertex normal, tangent and bitangent for each vertex in triangle = 18
// three lines for edges = 4 (due to usage of line strip output primitive)
// one line for normal of the triangle = 2
layout(line_strip, max_vertices = 24) out;

///////////////////////////////////////////////////////////////////

// data from vertex shader for three vertices of the triangle
in vec3	o_normal[];
in vec3	o_tangent[];
in vec3 o_bitangent[];

// matrices
uniform mat4 u_view_mat;
uniform mat4 u_proj_mat;

// modifier for size of the normals, tangents and bitangents
uniform float	u_normalScale;

// colors for different type of new lines
uniform vec4 u_edgeColor;
uniform vec4 u_faceNormalColor;
uniform vec4	u_normalColor;
uniform vec4	u_tangentColor;
uniform vec4	u_bitangentColor;

// color to fragment shader
out vec4 o_color;

///////////////////////////////////////////////////////////////////

void main()
{
   mat4 viewProjection = u_proj_mat * u_view_mat;

   // normals of each vertex of the triangle
   vec3 nor[3];
   nor[0] = o_normal[0].xyz;
   nor[1] = o_normal[1].xyz;
   nor[2] = o_normal[2].xyz;

   // positions of each vertex of the triangle
   // shifted a bit along normal
   // so there won't be Z fighting when rendered over the mesh
   vec4 pos[3];
   pos[0] = viewProjection * vec4(gl_in[0].gl_Position.xyz + nor[0] * 0.01, 1.0);
   pos[1] = viewProjection * vec4(gl_in[1].gl_Position.xyz + nor[1] * 0.01, 1.0);
   pos[2] = viewProjection * vec4(gl_in[2].gl_Position.xyz + nor[2] * 0.01, 1.0);

   // output normals, tangents and bitangents for each vertex of the triangle
   for(int i=0; i < gl_in.length(); i++)
   {
      // get position of the vertex
      vec3 P = gl_in[i].gl_Position.xyz;

      // create normal for vertex
      o_color = u_normalColor;
      gl_Position = pos[i];
      EmitVertex();
      gl_Position = viewProjection * vec4(P + o_normal[i].xyz 
                        * u_normalScale, 1.0);
      EmitVertex();
      EndPrimitive();

      // create tangent for vertex
      o_color = u_tangentColor;
      gl_Position = pos[i];
      EmitVertex();
      gl_Position = viewProjection * vec4(P + o_tangent[i].xyz * u_normalScale, 1.0);
      EmitVertex();
      EndPrimitive();

      // create bitangent for vertex
      o_color = u_bitangentColor;
      gl_Position = pos[i];
      EmitVertex();
      gl_Position = viewProjection * vec4(P + 
                              o_bitangent[i].xyz * u_normalScale, 1.0);
      EmitVertex();
      EndPrimitive();
   }

   // create edges for triangle
   o_color = u_edgeColor;
   gl_Position = pos[0];
   EmitVertex();
   gl_Position = pos[1];
   EmitVertex();
   gl_Position = pos[2];
   EmitVertex();
   gl_Position = pos[0];
   EmitVertex();
   // end line strip after four added vertices, so we will get three lines
   EndPrimitive();

   // create normal for triangle
   o_color = u_faceNormalColor;
   // form two vectors from triangle
   vec3 V0 = gl_in[0].gl_Position.xyz - gl_in[1].gl_Position.xyz;
   vec3 V1 = gl_in[2].gl_Position.xyz - gl_in[1].gl_Position.xyz;
   // calculate normal as perpendicular to two vectors of the triangle
   vec3 N = normalize(cross(V1, V0));
   // position as arithmetic average
   vec3 P = (gl_in[0].gl_Position.xyz + gl_in[1].gl_Position.xyz 
                        + gl_in[2].gl_Position.xyz) / 3.0;
   gl_Position = viewProjection * vec4(P, 1.0);
   EmitVertex();
   gl_Position = viewProjection * vec4(P + N * u_normalScale, 1.0);
   EmitVertex();
   EndPrimitive();
}