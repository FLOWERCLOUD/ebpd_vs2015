#version 120

varying vec4 position;

uniform vec3 dataStep;
uniform vec3 world_size;
uniform vec3 world_start;

uniform sampler3D dataFieldTex;

const vec3 diffuseMaterial = vec3(0.7, 0.7, 0.7);
const vec3 specularMaterial = vec3(0.99, 0.99, 0.99);
const vec3 ambiantMaterial = vec3(0.1, 0.1, 0.1);

float fetch_field( vec3 pos )
{
    return texture3D(dataFieldTex, (pos - world_start) / world_size ).a;
}

void main(void)
{

#if 1
	
    vec3 step = dataStep * 4.; // Hack large steps will smooth out imperfections ^^
    vec3 grad = vec3( fetch_field(position.xyz + vec3(step.x, 0, 0)) - fetch_field(position.xyz + vec3(-step.x, 0, 0)),
                      fetch_field(position.xyz + vec3(0, step.y, 0)) - fetch_field(position.xyz + vec3(0, -step.y, 0)),
		              fetch_field(position.xyz + vec3(0, 0, step.z)) - fetch_field(position.xyz + vec3(0, 0, -step.z)) );
        
        
    vec3 lightVec=normalize( world_size/*gl_LightSource[0].position.xyz*/ - position.xyz);
    
    vec3 normalVec = normalize(grad);

	vec3 color=gl_Color.rgb*0.5+abs(normalVec)*0.5;

    // calculate half angle vector
    vec3 eyeVec = vec3(0.0, 0.0, 1.0);
    vec3 halfVec = normalize(lightVec + eyeVec);
    
    // calculate diffuse component
    vec3 diffuse = vec3(abs(dot(normalVec, lightVec))) * color*diffuseMaterial;
    // Add a diffuse light
    vec3 diffuse2 = vec3(abs(dot(normalVec, normalize( world_start - position.xyz)))) * color*diffuseMaterial;
    vec3 diffuse3 = vec3(abs(dot(normalVec, normalize( world_start + vec3(world_size.xy * 0.5, 0.0) - position.xyz)))) * color*diffuseMaterial;

    // calculate specular component
    vec3 specular = vec3(abs(dot(normalVec, halfVec)));
    specular = pow(specular.x, 32.0) * specularMaterial;
    
    // combine diffuse and specular contributions and output final vertex color
    gl_FragColor.rgb =gl_Color.rgb*ambiantMaterial + diffuse + diffuse2 + diffuse3 + specular;
    gl_FragColor.a = 1.0;
    
#else

	gl_FragColor=gl_Color;
#endif

    //float iso = fetch_field(position.xyz)* 4. - 1.9;
    //gl_FragColor= vec4(iso,iso,iso, 1.0);
    
    //gl_FragColor= vec4((normalVec+1.0)*0.5, 1.0);    
}
