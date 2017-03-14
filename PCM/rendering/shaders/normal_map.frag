const float far  = 100.;
const float near = 1.;

#if 0
varying in vec3 normal;



void main(void)
{
    vec3 normal_n = normalize(normal);
    float depth   = gl_FragCoord.z;//(2.0 * near) / (far + near - gl_FragCoord.z * (far-near));
    gl_FragColor  = vec4(normal_n, depth);
}
#endif



varying vec3 Normal;
varying float depth;
void main( void )
{
    //float depth_   = (2.0 * near) / (far + near - gl_FragCoord.z * (far-near));
   gl_FragColor = vec4(normalize(Normal),depth);
   //gl_FragColor = vec4(depth,depth,depth, 1.0);
}
