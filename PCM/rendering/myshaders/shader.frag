#version 330 core

in vec2 TexCoords;
in vec4 ObjectColor;
in vec3 FragPos;  
in vec3 Normal;

out vec4 color;

uniform vec3 lightPos; 
uniform vec3 viewPos;
uniform vec3 lightColor;

uniform sampler2D texture_diffuse1;

void main()
{    
 //   color = texture(texture_diffuse1, TexCoords);
//      color = vec4(1.0f,0.0f,0.0f,1.0f);
      // Ambient
    float ambientStrength = 0.1f;
    vec3 ambient = ambientStrength * lightColor;
  	
    // Diffuse 
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Specular
    float specularStrength = 0.5f;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;  
    
	vec3 tempobjcolor = vec3(ObjectColor[0],ObjectColor[1],ObjectColor[2]);
    
    vec3 result = (ambient + diffuse + specular) * tempobjcolor;

    color = vec4(result,ObjectColor[3]);
}