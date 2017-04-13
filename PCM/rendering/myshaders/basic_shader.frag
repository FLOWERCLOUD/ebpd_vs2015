#version 330 core


out vec4 color;

uniform vec3 lightPos; 
uniform vec3 viewPos;
uniform vec3 lightColor;


void main()
{    
    color = vec4(0.4f,0.4f,0.785f,1.0f);
}