/*input.hpp*/
#ifndef INPUT_HPP
#define INPUT_HPP

#include <string>


template <typename T>
void readfile(const std::string & filename, 
				unsigned int & nx, unsigned int & ny, unsigned int & nz, unsigned int & nt, 
				T & dx, T & dt,
				T & a_2, T & a_4, T & M, T & K, 
				unsigned int & t_skip);

#endif