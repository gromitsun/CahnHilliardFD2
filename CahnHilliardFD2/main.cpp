//
//  main.cpp
//  AllenCahnFD
//
//  Created by Yue Sun on 7/17/15.
//  Copyright (c) 2015 Yue Sun. All rights reserved.
//

#include <iostream>
#include <cstring>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "simulator_2d.hpp"
#include "simulator_3d.hpp"

int main(int argc, const char * argv[])
{
    // insert code here...
    std::cout << "Hello, World!\n";
    
    if (argc < 2)
    {
        std::cerr << "ERROR: No input file specified!" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    int sim_dim;
    
    if (argc > 2)
        if (strcmp(argv[2],"-2")==0)
            sim_dim = 2;
    
    if (sim_dim == 2)
    {
        Simulator_2D<float> sim{};
        
        // Read input parameters
        sim.read_input(argv[1]);
        
        
        sim.init_cl(CL_DEVICE_TYPE_GPU, 1);
        
        sim.build_kernel("kernel_float_2d.cl");
        
        sim.init_sim(0, 0.001);
        
        sim.run();
        
        return 0;
    }

    Simulator_3D<float> sim{};
    
    // Read input parameters
    sim.read_input(argv[1]);

    
    sim.init_cl(CL_DEVICE_TYPE_GPU, 1);
    
    sim.build_kernel("kernel_float_3d.cl");
    
    sim.init_sim(0, 0.001);
    
    sim.run();
    
    return 0;
}
