#ifndef LOAD_OBJ
#define LOAD_OBJ

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <glm/glm.hpp>

std::pair <std::vector <float>, std::vector <unsigned int> > load_obj(std::string path);

#endif
