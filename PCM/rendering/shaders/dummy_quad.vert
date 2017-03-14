attribute vec2 vertex_coord;

varying vec2 tex_coord;

void main(){
	
		tex_coord = vertex_coord*0.5 + (0.5);
		gl_Position = vec4(vertex_coord,0.,1.);
}
