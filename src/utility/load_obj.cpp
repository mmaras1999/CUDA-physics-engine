#include "load_obj.hpp"
#include <iostream>

std::pair <std::vector <float>, std::vector <unsigned int> > load_obj(std::string path)
{
    struct Triangle
    {
        int id_x, id_y, id_z;
    };

    std::vector <float> vertices;
    std::vector <unsigned int> indices;

    std::ifstream file(path);
    std::string s;

    while(std::getline(file, s))
    {
        std::stringstream ss(s);

        std::string type;
    
        ss >> type;

        if (type == "v")
        {
            float x, y, z;

            ss >> x >> y >> z;
            vertices.emplace_back(x);
            vertices.emplace_back(y);
            vertices.emplace_back(z);
        }
        else if (type == "f")
        {
            std::vector <std::string> read_strings;

            while(ss.rdbuf()->in_avail())
            {
                std::string tmp;
                ss >> tmp;

                if(tmp.size())
                    read_strings.push_back(tmp);
            }
            
            if (read_strings.size() <= 3)
            {
                for (int i = 0; i < 3; ++i)
                {
                    indices.emplace_back(std::stoi(read_strings[i].substr(0, read_strings[i].find('/'))));
                }  
            }
            else
            {
                for (int i = 0; i < 3; ++i)
                {
                    indices.emplace_back(std::stoi(read_strings[i].substr(0, read_strings[i].find('/'))));
                }

                indices.emplace_back(std::stoi(read_strings[0].substr(0, read_strings[0].find('/'))));
                indices.emplace_back(std::stoi(read_strings[2].substr(0, read_strings[2].find('/'))));
                indices.emplace_back(std::stoi(read_strings[3].substr(0, read_strings[3].find('/'))));
            }
        }
    }

    file.close();

    std::cout << "Object vertices: " << vertices.size() / 3 << ", triangles: " << indices.size() / 3 << std::endl;

    return {vertices, indices};
}