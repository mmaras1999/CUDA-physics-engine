#include "particle_sim_app.hpp"
#include "object_viewer_app.hpp"
#include "rigid_body_app.hpp"
#include "utility/random_generator.hpp"

#include <iostream>
#include <filesystem>
#include <string>
#include <set>
#include <map>

std::pair <std::vector <std::string>, std::map <std::string, std::string> > parse_args(int argc, char* argv[])
{
    std::set <std::string> available_flags = {
        "--help",
        "--obj_viewer",
        "--particles_only",
        "--seed"
    };

    std::map <std::string, std::string> parsed_args;
    std::set <std::string> no_value_flags{
        "--help",
        "--obj_viewer",
        "--particles_only"
    };

    std::vector <std::string> unnamed_args;

    std::string arg_name = "";

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg_name == "")
        {
            if (!available_flags.count(arg))
            {
                unnamed_args.push_back(arg);
                continue;
            }

            if (no_value_flags.count(arg))
                parsed_args[arg] = "";
            else
                arg_name = arg;
        }
        else
        {
            parsed_args[arg_name] = arg;
            arg_name = "";
        }
    }

    return {unnamed_args, parsed_args};
}

int main(int argc, char* argv[])
{
    if (!ALLOW_RESIZE)
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    auto [unnamed_args, parsed_args] = parse_args(argc, argv);
    
    if (parsed_args.count("--help"))
    {
        std::cout << R"(Usage:)" "\n"
                     R"(./particle_simulator <object_path:string> <start_cfg_path:string> - object_path is the path to the .obj file, start_cfg_path is the path to the .cfg file with starting configuration.)" "\n"
                     R"(Available optional arguments:)" "\n"
                     R"( --seed <seed_value:int> - sets random generator's seed,)" "\n"
                     R"( --obj_viewer - opens object viewer app,)" "\n"
                     R"( --particles_only - simulates particles independently.)"
        << std::endl;

        return 0;
    }

    std::string object_path = "";
    std::string start_config_path = "";

    if (unnamed_args.size() < 2)
    {
        std::cout << "Invalid arguments! See --help for usage." << std::endl;
    }
    else
    {
        object_path = unnamed_args[0];
        start_config_path = unnamed_args[1];

        if(!std::filesystem::exists(object_path))
        {
            std::cout << "Error: invalid .obj path!" << std::endl;
            return 1;
        }

        if(!std::filesystem::exists(start_config_path))
        {
            std::cout << "Error: invalid config path!" << std::endl;
            return 1;
        }
    }

    if (parsed_args.count("--seed"))
    {
        try
        {
            int seed = std::stoi(parsed_args["--seed"]);
            std::cout << "Seed set to " << seed << std::endl;
            RandomGenerator::getInstance().set_seed(seed);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            return 1;
        }
    }

    if (parsed_args.count("--obj_viewer"))
    {
        ObjectViewerApp app{1920, 1080, "Obj Viewer by Michal Maras", 0, object_path};
        app.run();
        
        return 0;
    }

    if (parsed_args.count("--particles_only"))
    {
        ParticleSimApp app{1920, 1080, "Particle Simulator by Michal Maras", 0, object_path, start_config_path};
        app.run();

        return 0;
    }

    RigidBodyApp app{1920, 1080, "Rigid Body Simulator by Michal Maras", 0, object_path, start_config_path};
    app.run();

    return 0;
}
