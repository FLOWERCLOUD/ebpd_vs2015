#version 330

// color from geometry shader
in vec4 o_color;

// color to the framebuffer
out vec4	resultingColor;

///////////////////////////////////////////////////////////////////

void main(void)
{
   // output color to the framebuffer
   resultingColor = o_color;
}