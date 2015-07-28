#define __NINETEEN_STENCIL__
#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;


inline float4 laplacian_3d(__read_only image3d_t Phi,
                           float4 phi,
                           const float dx,
                           float4 normalizedCoord,
                           int4 size)
{
    // Get increments in x,y,z
    float incrementx = 1.0f/size.x;
    float incrementy = 1.0f/size.y;
    float incrementz = 1.0f/size.z;
    
    // Calculate stencils
    float4 xm= (read_imagef(Phi,sampler,(normalizedCoord+(float4){-incrementx,0,0,0})));
    float4 xp= (read_imagef(Phi,sampler,(normalizedCoord+(float4){incrementx,0,0,0})));
    float4 ym= (read_imagef(Phi,sampler,(normalizedCoord+(float4){0,-incrementy,0,0})));
    float4 yp= (read_imagef(Phi,sampler,(normalizedCoord+(float4){0,incrementy,0,0})));
    float4 zm= (read_imagef(Phi,sampler,(normalizedCoord+(float4){0,0,-incrementz,0})));
    float4 zp= (read_imagef(Phi,sampler,(normalizedCoord+(float4){0,0,incrementz,0})));
#ifndef __NINETEEN_STENCIL__
    return (xm+xp+ym+yp+zm+zp-6.0f*phi)/(dx*dx);  // 7-point stencil
#else
    float4 xym= (read_imagef(Phi,sampler,(normalizedCoord+(float4){-incrementx,-incrementy,0,0})));
    float4 xyp= (read_imagef(Phi,sampler,(normalizedCoord+(float4){incrementx,incrementy,0,0})));
    float4 xmyp= (read_imagef(Phi,sampler,(normalizedCoord+(float4){-incrementx,incrementy,0,0})));
    float4 xpym= (read_imagef(Phi,sampler,(normalizedCoord+(float4){incrementx,-incrementy,0,0})));
    
    float4 xzm= (read_imagef(Phi,sampler,(normalizedCoord+(float4){-incrementx,0,-incrementz,0})));
    float4 xzp= (read_imagef(Phi,sampler,(normalizedCoord+(float4){incrementx,0,incrementz,0})));
    float4 xmzp= (read_imagef(Phi,sampler,(normalizedCoord+(float4){-incrementx,0,incrementz,0})));
    float4 xpzm= (read_imagef(Phi,sampler,(normalizedCoord+(float4){incrementx,0,-incrementz,0})));
    
    float4 yzm= (read_imagef(Phi,sampler,(normalizedCoord+(float4){0,-incrementy,-incrementz,0})));
    float4 yzp= (read_imagef(Phi,sampler,(normalizedCoord+(float4){0,incrementy,incrementz,0})));
    float4 ymzp= (read_imagef(Phi,sampler,(normalizedCoord+(float4){0,-incrementy,incrementz,0})));
    float4 ypzm= (read_imagef(Phi,sampler,(normalizedCoord+(float4){0,incrementy,-incrementz,0})));
    
    return ((xm+xp+ym+yp+zm+zp)/3.0f
            + (xyp+xym+xpym+xmyp+xzm+xzp+xmzp+xpzm+yzm+yzp+ymzp+ypzm)/6.0f
            - 4.0f*phi)/(dx*dx); // 19-point stencil
#undef __NINETEEN_STENCIL__
#endif
}


__kernel void brac_3d(__read_only image3d_t Phi,
                      __write_only image3d_t Bracket,
                      const float a_2,
                      const float a_4,
                      const float K,
                      const float dx)
{
    // Get pixel coordinates
    int4 coord = {get_global_id(0),get_global_id(1),get_global_id(2),0};
    int4 size = {get_global_size(0),get_global_size(1),get_global_size(2),0};
    float4 normalizedCoord = (float4)((float)coord.x/size.x, (float)coord.y/size.y, (float)coord.z/size.z, 0);
    
    // Read in Phi
    float4 phi = (read_imagef(Phi, sampler, normalizedCoord).x);
    
    // Calculate Laplacian of Phi
    float4 laplacian = laplacian_3d(Phi, phi, dx, normalizedCoord, size);
    
    // Calculate terms in bracket
    float4 bracket = - 2.0f * K * laplacian + a_2 * phi + a_4 * phi * phi * phi;
    
    // Write result to memory object
    write_imagef(Bracket,coord,bracket);
}


__kernel void step_3d(__read_only image3d_t Phi,
                      __read_only image3d_t Bracket,
                      __write_only image3d_t PhiNext,
                      const float M,
                      const float dx,
                      const float dt)
{
    // Get pixel coordinates
    int4 coord = {get_global_id(0),get_global_id(1),get_global_id(2),0};
    int4 size = {get_global_size(0),get_global_size(1),get_global_size(2),0};
    float4 normalizedCoord = (float4)((float)coord.x/size.x, (float)coord.y/size.y, (float)coord.z/size.z, 0);
    
    // Read in Phi & Bracket
    float4 phi = (read_imagef(Phi, sampler, normalizedCoord).x);
    float4 bracket = (read_imagef(Bracket, sampler, normalizedCoord).x);
    
    // Calculate Laplacian of Bracket
    float4 laplacian = laplacian_3d(Bracket, bracket, dx, normalizedCoord, size);
    
    // Make one step forward
    float4 phi_next = phi + dt * M * laplacian;
    
    // Write result to memory object
    write_imagef(PhiNext,coord,phi_next);
}

