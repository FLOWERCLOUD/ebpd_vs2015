#version 330 core

out vec4 FragColor;

in VS_OUT {
    vec3 FragPos;
    vec2 TexCoords;
	vec3 Normal
	vec4 ObjectColor;
    vec3 TangentLightPos;
    vec3 TangentViewPos;
    vec3 TangentFragPos;
} fs_in;

uniform vec3 lightPos; 
uniform vec3 viewPos;
uniform vec3 lightColor;

uniform sampler2D texture_ambient1;
uniform sampler2D texture_diffuse1;
uniform sampler2D texture_specular;
uniform sampler2D texture_normal1;
uniform sampler2D texture_height1;


uniform bool use_texture_diffuse1;
uniform bool use_texture_specular1;
uniform bool use_texture_normal1;
uniform bool use_texture_ambient1;

void main()
{    
	if(use_texture_diffuse1)
	{
		vec3 normal;
		// Obtain normal from normal map in range [0,1]
		if(use_texture_normal1)
			normal = texture(texture_normal1, fs_in.TexCoords).rgb;
		else
			normal = fs_in.Normal;
			
		// Transform normal vector to range [-1,1]
		normal = normalize(normal * 2.0 - 1.0);  // this normal is in tangent space

		// Get diffuse color
		vec3 color = texture(texture_diffuse1, fs_in.TexCoords).rgb;
		// Ambient
		vec3 ambient;
		if(use_texture_ambient1)
		{
			ambient = texture(texture_ambient1, fs_in.TexCoords).rgb;
		}else
		{
			ambient = 0.1 * color;
		}
		
		// Diffuse
		vec3 lightDir = normalize(fs_in.TangentLightPos - fs_in.TangentFragPos);
		float diff = max(dot(lightDir, normal), 0.0);
		vec3 diffuse = diff * color;
		// Specular
		vec3 viewDir = normalize(fs_in.TangentViewPos - fs_in.TangentFragPos);
		vec3 reflectDir = reflect(-lightDir, normal);
		vec3 halfwayDir = normalize(lightDir + viewDir);  
		float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
		vec3 specular;
		if(use_texture_specular1)
		{
			specular = texture(texture_specular1, fs_in.TexCoords).rgb;
		}	
		else
		{
			specular= vec3(0.2) * spec;
		}
		
		FragColor = vec4(ambient + diffuse + specular, 1.0f);
		
		
	}
	else
    {
			  // Ambient
		float ambientStrength = 0.1f;
		vec3 ambient = ambientStrength * lightColor;
		
		// Diffuse 
		vec3 norm = normalize(Normal);
		vec3 lightDir = normalize(lightPos - FragPos);
	   float diff = max(dot(norm, lightDir), 0.0);
	//	float diff = max(dot(norm, lightDir), -dot(norm, lightDir)); //enable backface render
		vec3 diffuse = diff * lightColor;
		
		// Specular
		float specularStrength = 0.5f;
		vec3 viewDir = normalize(viewPos - FragPos);
		vec3 reflectDir = reflect(-lightDir, norm);  
		float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
		vec3 specular = specularStrength * spec * lightColor;  
		
		vec3 tempobjcolor = vec3(ObjectColor[0],ObjectColor[1],ObjectColor[2]);
		
		vec3 result = (ambient + diffuse + specular) * tempobjcolor;

		FragColor = vec4(result,ObjectColor[3]);
	
	}

}