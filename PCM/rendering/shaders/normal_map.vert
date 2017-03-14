#if 0

varying out vec3 normal;

void main(void)
{
    //calcul de la normale en espace camera
    normal = vec3(gl_NormalMatrix * gl_Normal);
    //calcul de la position du vertex en espace ecran
    gl_Position =  ftransform();
}

#endif

varying vec3 Normal;
varying float depth; // in eye space
void main( void )
{
    vec4 viewPos = gl_ModelViewMatrix * gl_Vertex;
    depth = -viewPos.z/100.0;
    //depth = -length(viewPos)/1000.0;

   gl_Position = ftransform();
   //depth = gl_Position.z/1000.0; // divided by 1000 for a better range while testing, should be changed
  // normal in eye space
   Normal      = normalize(( gl_ModelViewMatrix * vec4( gl_Normal.xyz, 0.0 ) ).xyz);
}
