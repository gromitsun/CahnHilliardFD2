//
//  simulator_2d.h
//  AllenCahnFD
//
//  Created by Yue Sun on 7/20/15.
//  Copyright (c) 2015 Yue Sun. All rights reserved.
//

#ifndef __AllenCahnFD__simulator_2d__
#define __AllenCahnFD__simulator_2d__

#include "simulator.hpp"

template <typename T>
class Simulator_2D : public Simulator<T>
{
private:
    // physical parameters
    T _a_2;
    T _a_4;
    T _M;
    T _K;
    // simulation parameters
    T _dx;
    T _dt;
    unsigned int _nt;
    unsigned int _t_skip;
    
    cl_mem _img_Phi;
    cl_mem _img_Bracket;
    cl_mem _img_PhiNext;
    
    cl_mem _rotate_var;
    
    cl_kernel _kernel_brac_2d;
    cl_kernel _kernel_step_2d;
    
    size_t _local_size[3];
    size_t _global_size[3];
    
public:
    Simulator_2D();
    Simulator_2D(const unsigned int & nx,
                 const unsigned int & ny);
    Simulator_2D(const unsigned int & nx,
                 const unsigned int & ny,
                 const T & a_2,
                 const T & a_4,
                 const T & M,
                 const T & K,
                 const unsigned int & t_skip);
    ~Simulator_2D();
    
    void read_input(const char * filename);
    
    cl_int build_kernel(const char * kernel_file="kernel_float.cl");
    void init_sim(const T & mean, const T & sigma);
    cl_int write_mem();
    cl_int read_mem();
    
    void step(const T & dt);
    void steps(const T & dt, const unsigned int & nsteps, const bool finish=true, const bool cputime=true);
    
    void run();
};


#endif /* defined(__AllenCahnFD__simulator_2d__) */
