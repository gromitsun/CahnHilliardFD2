#define __NINE_STENCIL__

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;



inline float4 laplacian_2d(__read_only image2d_t Phi,
                           float4 phi,
                           const float dx,
                           float2 normalizedCoord,
                           int2 size)
{
    // Get increments in x,y
    float incrementx = 1.0f/size.x;
    float incrementy = 1.0f/size.y;
    
    // Calculate stencils
    float4 xm= (read_imagef(Phi,sampler,(normalizedCoord+(float2){-incrementx,0})));
    float4 xp= (read_imagef(Phi,sampler,(normalizedCoord+(float2){incrementx,0})));
    float4 ym= (read_imagef(Phi,sampler,(normalizedCoord+(float2){0,-incrementy})));
    float4 yp= (read_imagef(Phi,sampler,(normalizedCoord+(float2){0,incrementy})));
#ifndef __NINE_STENCIL__
    return (xm+xp+ym+yp-4.0f*phi)/(dx*dx);  // 5-point stencil
#else
    float4 xym= (read_imagef(Phi,sampler,(normalizedCoord+(float2){-incrementx,-incrementy})));
    float4 xyp= (read_imagef(Phi,sampler,(normalizedCoord+(float2){incrementx,incrementy})));
    float4 xpym= (read_imagef(Phi,sampler,(normalizedCoord+(float2){incrementx,-incrementy})));
    float4 xmyp= (read_imagef(Phi,sampler,(normalizedCoord+(float2){-incrementx,incrementy})));
    
    return ((xm+xp+ym+yp)/2.0f + (xyp+xym+xmyp+xpym)/4.0f - 3.0f*phi)/(dx*dx); // 9-point stencil
#undef __NINE_STENCIL__
#endif
}



__kernel void brac_2d(__read_only image2d_t Phi,
					  __write_only image2d_t Bracket,
					  const float a_2,
					  const float a_4,
					  const float K,
					  const float dx)
{
	// Get pixel coordinates
    int2 coord = {get_global_id(0),get_global_id(1)};
	int2 size = {get_global_size(0),get_global_size(1)};
    float2 normalizedCoord = (float2)((float)coord.x/size.x, (float)coord.y/size.y);
//    float2 normalizedCoord = (float2)((coord.x+0.5f)/size.x, (coord.y+0.5f)/size.y);
    
    // Read in Phi
    float4 phi = (read_imagef(Phi, sampler, normalizedCoord).x);
    
    // Calculate Laplacian of Phi
    float4 laplacian = laplacian_2d(Phi, phi, dx, normalizedCoord, size);
    
    // Calculate terms in bracket
    float4 bracket = - 2.0f * K * laplacian + a_2 * phi + a_4 * phi * phi * phi;
    
    // Write result to memory object
    write_imagef(Bracket,coord,bracket);
}


__kernel void step_2d(__read_only image2d_t Phi,
                      __read_only image2d_t Bracket,
                      __write_only image2d_t PhiNext,
                      const float M,
                      const float dx,
                      const float dt)
{
    // Get pixel coordinates
    int2 coord = {get_global_id(0),get_global_id(1)};
    int2 size = {get_global_size(0),get_global_size(1)};
    float2 normalizedCoord = (float2)((float)coord.x/size.x, (float)coord.y/size.y);
//    float2 normalizedCoord = (float2)((coord.x+0.5f)/size.x, (coord.y+0.5f)/size.y);
    
    // Read in Phi & Bracket
    float4 phi = (read_imagef(Phi, sampler, normalizedCoord).x);
    float4 bracket = (read_imagef(Bracket, sampler, normalizedCoord).x);
   
    // Calculate Laplacian of Bracket
    float4 laplacian = laplacian_2d(Bracket, bracket, dx, normalizedCoord, size);
    
    // Make one step forward
    float4 phi_next = phi + dt * M * laplacian;
    
    // Write result to memory object
    write_imagef(PhiNext,coord,phi_next);
}