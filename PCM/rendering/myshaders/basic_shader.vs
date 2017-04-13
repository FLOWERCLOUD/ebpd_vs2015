#version 330 core
layout (location = 0) in vec3 position;

//uniform mat4 model;
//uniform mat4 view;
//uniform mat4 projection;
uniform mat4 allmatrix;

void main()
{
    gl_Position = allmatrix * vec4(position, 1.0f);
}