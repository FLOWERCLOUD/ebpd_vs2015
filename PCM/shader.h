#pragma once
#include <CustomGL\glew.h>
#include <fstream>
#include <sstream>
#include <iostream>
class Shader
{
public:
	GLuint Program;
	// Constructor generates the shader on the fly
	Shader(const GLchar* vertexPath, const GLchar* fragmentPath, const GLchar* geometryPath = nullptr);
	// Uses the current shader
	void Use() { glUseProgram(this->Program); }
	void UnUse() { glUseProgram(NULL); }

private:
	void checkCompileErrors(GLuint shader, std::string type);
};