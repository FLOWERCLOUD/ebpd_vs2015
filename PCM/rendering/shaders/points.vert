//varying out vec3 vnormal;
//varying out vec4 vposition;


void main(void)
{
    //calcul de la position du vertex en espace camera
    //vposition =  gl_ModelViewMatrix *  gl_Vertex;
    //calcul de la normale en espace camera
    //vnormal = vec3(gl_NormalMatrix * gl_Normal);
    //calcul de la position du vertex en espace ecran
    gl_Position =  ftransform();
}
