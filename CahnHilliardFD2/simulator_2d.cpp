//
//  simulator_2d.cpp
//  AllenCahnFD
//
//  Created by Yue Sun on 7/20/15.
//  Copyright (c) 2015 Yue Sun. All rights reserved.
//

#include <cstdio>
#include <iostream>

#include "simulator_2d.hpp"
#include "cl_common.hpp"
#include "input.hpp"
#include "randn.hpp"



template <typename T>
Simulator_2D<T>::Simulator_2D() : Simulator<T>(0, 0, 0, 0) {}


template <typename T>
Simulator_2D<T>::Simulator_2D(const unsigned int & nx,
                              const unsigned int & ny) : Simulator<T>(nx, ny, 1, 2) {}


template <typename T>
Simulator_2D<T>::Simulator_2D(const unsigned int & nx,
                              const unsigned int & ny,
                              const T & a_2,
                              const T & a_4,
                              const T & M,
                              const T & K,
                              const unsigned int & t_skip) : Simulator<T>(nx, ny, 1, 2)
{
    _a_2 = a_2;
    _a_4 = a_4;
    _M = M;
    _K = K;
    _t_skip = t_skip;
}


template <typename T>
Simulator_2D<T>::~Simulator_2D()
{
    if (Simulator<T>::_cl_initialized)
    {
        clReleaseKernel(_kernel_brac_2d);
        clReleaseKernel(_kernel_step_2d);
        clReleaseMemObject(_img_Phi);
        clReleaseMemObject(_img_Bracket);
        clReleaseMemObject(_img_PhiNext);
        clReleaseMemObject(_rotate_var);
    }
    
    delete [] Simulator<T>::_data;
}



template <typename T>
void Simulator_2D<T>::read_input(const char * filename)
{
    readfile(filename, Simulator<T>::_dim.x, Simulator<T>::_dim.y, Simulator<T>::_dim.z,
             _nt, _dx, _dt, _a_2, _a_4, _M, _K, _t_skip);
    Simulator<T>::_size = Simulator<T>::_dim.x * Simulator<T>::_dim.y;
}


template <typename T>
cl_int Simulator_2D<T>::build_kernel(const char * kernel_file)

{
    cl_int err;
    cl_program program;

    // Build program with source (filename) on device+context
    CHECK_RETURN_N(program, CreateProgram(Simulator<T>::context, Simulator<T>::device, kernel_file, err), err);
    CHECK_RETURN_N(_kernel_brac_2d, clCreateKernel(program, "brac_2d", &err), err);
    CHECK_RETURN_N(_kernel_step_2d, clCreateKernel(program, "step_2d", &err), err);

    //Create Images;
   
    cl_image_desc r_desc;
    r_desc.image_type=CL_MEM_OBJECT_IMAGE2D;
    r_desc.image_width=Simulator<T>::_dim.x;
    r_desc.image_height=Simulator<T>::_dim.y;
    r_desc.image_depth=1;
    r_desc.image_row_pitch=0;
    r_desc.image_slice_pitch=0;
    r_desc.num_mip_levels=0;
    r_desc.num_samples=0;
    r_desc.buffer=NULL;
    
    cl_image_format fmt;
    fmt.image_channel_data_type=CL_FLOAT; /* DOES NOT WORK WITH DOUBLE */
    fmt.image_channel_order=CL_R;
    
    T *v=(T *)calloc(Simulator<T>::_size,sizeof(T));
    for(int i=0;i<Simulator<T>::_size;i++)
    {
        v[i]=rand()%101/(T)100;
    }
    CHECK_RETURN_N(_img_Phi,clCreateImage(Simulator<T>::context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, &fmt, &r_desc, v, &err),err)
    CHECK_RETURN_N(_img_Bracket,clCreateImage(Simulator<T>::context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, &fmt, &r_desc, v, &err),err)
    CHECK_RETURN_N(_img_PhiNext,clCreateImage(Simulator<T>::context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, &fmt, &r_desc, v, &err),err)
    
    CHECK_ERROR(clSetKernelArg(_kernel_brac_2d, 0, sizeof(cl_mem), &_img_Phi))
    CHECK_ERROR(clSetKernelArg(_kernel_brac_2d, 1, sizeof(cl_mem), &_img_Bracket))
    CHECK_ERROR(clSetKernelArg(_kernel_brac_2d, 2, sizeof(T), &_a_2))
    CHECK_ERROR(clSetKernelArg(_kernel_brac_2d, 3, sizeof(T), &_a_4))
    CHECK_ERROR(clSetKernelArg(_kernel_brac_2d, 4, sizeof(T), &_K))
    CHECK_ERROR(clSetKernelArg(_kernel_brac_2d, 5, sizeof(T), &_dx))
    
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 0, sizeof(cl_mem), &_img_Phi))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 1, sizeof(cl_mem), &_img_Bracket))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 2, sizeof(cl_mem), &_img_PhiNext))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 3, sizeof(T), &_M))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 4, sizeof(T), &_dx))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 5, sizeof(T), &_dt))
    
    free(v);
    
    _local_size[0]=8;
    _local_size[1]=8;
    _local_size[2]=1;
    
    _global_size[0]=Simulator<T>::_dim.x;
    _global_size[1]=Simulator<T>::_dim.y;
    _global_size[2]=1;
    
    Simulator<T>::_cl_initialized = true;
    
    return CL_SUCCESS;
}


template <typename T>
cl_int Simulator_2D<T>::write_mem()
{
    CHECK_ERROR(this->WriteArray(_img_Phi, Simulator<T>::_data, _global_size[0], _global_size[1], _global_size[2]));
    return CL_SUCCESS;
}


template <typename T>
cl_int Simulator_2D<T>::read_mem()
{
    CHECK_ERROR(this->ReadArray(_img_Phi, Simulator<T>::_data, _global_size[0], _global_size[1], _global_size[2]));
    return CL_SUCCESS;
}


template <typename T>
void Simulator_2D<T>::init_sim(const T & mean, const T & sigma)
{
    if (Simulator<T>::_data == NULL)
        Simulator<T>::_data = new T[Simulator<T>::_size];
    
    gauss(Simulator<T>::_data, Simulator<T>::_size, mean, sigma);
    
    write_mem();
}


template <typename T>
void Simulator_2D<T>::steps(const T & dt, const unsigned int & nsteps, const bool finish, const bool cputime)
{
    if ((&dt != &_dt) && (dt != _dt))
    {
        _dt = dt;
        CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d, 5, sizeof(T), &_dt))
    }
    
    // Variables for timing
    ttime_t t1=0, t2=0, t3=0;
    double s=0, s2=0;
    
    if(cputime)
        t1= getTime();
    
    for (unsigned int t=0; t<nsteps; t++)
    {
        // Calculate terms in the bracket
        CHECK_ERROR_EXIT(clEnqueueNDRangeKernel(Simulator<T>::queue, _kernel_brac_2d, 2, NULL, _global_size, _local_size, 0, NULL, NULL));
        
        // Calculate laplacian of the bracket and do the time-stepping
        CHECK_ERROR_EXIT(clEnqueueNDRangeKernel(Simulator<T>::queue, _kernel_step_2d, 2, NULL, _global_size, _local_size, 0, NULL, NULL));
        
        //ROTATE VARIABLES
        _rotate_var=_img_Phi;
        _img_Phi=_img_PhiNext;
        _img_PhiNext=_rotate_var;
        
        CHECK_ERROR_EXIT(clSetKernelArg(_kernel_brac_2d, 0, sizeof(cl_mem), &_img_Phi));
        CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d, 2, sizeof(cl_mem), &_img_PhiNext));
    }
    
    if(cputime)
    {
        t2=getTime();
        s = subtractTimes(t2,t1);
    }
    
    if(finish)
        clFinish(Simulator<T>::queue);
    
    if(cputime)
    {
        t3=getTime();
        s2 = subtractTimes(t3,t1);
    }
    
    //this->_disp_mem=img_Phi;
    Simulator<T>::steps(_dt, nsteps);
    if(cputime)
        printf("It took %lfs to submit and %lfs to complete (%lfs per loops)\n",s,s2,s2/nsteps);
}


template <typename T>
void Simulator_2D<T>::run()
{
    if(Simulator<T>::current_step==0)
        Simulator<T>::writefile();
    
    ttime_t t0, t1;
    
    t0 = getTime();
    
    // for(unsigned int t=0; t < _nt; t+=_t_skip)
    for(unsigned int i=0; i < _nt/_t_skip; i++)
    {
        steps(_dt, _t_skip);
        read_mem();
        
        Simulator<T>::writefile();
    }
    
    t1 = getTime();
    
    std::cout << "Total computation time: " << subtractTimes(t1, t0) << "s." << std::endl;
    
}



/* Explicit instantiation */
template class Simulator_2D<float>;
//template void Simulator_2D<float>::read_input(const char *);
//template void Simulator_2D<float>::init_sim(const float &, const float &);
//template Simulator_2D<float>::Simulator_2D();

template class Simulator_2D<double>;










