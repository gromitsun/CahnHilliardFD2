/*input.cpp*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "input.hpp"

inline std::string trim_comment(const std::string & s, const std::string & delimiter="#")
{
    if (s.empty())
        return s;
    else
        return s.substr(0, s.find(delimiter));
}

inline std::string trim_right(
  const std::string & s,
  const std::string & delimiters = " \f\n\r\t\v" )
{
	if (s.empty())
		return s;
	return s.substr(0, s.find_last_not_of(delimiters) + 1);
}

inline std::string trim_left(
  const std::string & s,
  const std::string & delimiters = " \f\n\r\t\v" )
{
	if (s.empty())
		return s;
	return s.substr(s.find_first_not_of(delimiters));
}

inline std::string trim(
  const std::string & s,
  const std::string & delimiters = " \f\n\r\t\v" )
{
	return trim_left(trim_right(trim_comment(s), delimiters), delimiters);
}

inline bool parse_parameter(const std::string & line, std::string & key, std::string & value, const std::string & sep = "=")
{
	std::size_t pos = line.find(sep);
	if (pos == std::string::npos)
		return false;
	else
	{
		key = trim(line.substr(0, pos));
		value = trim(line.substr(pos+1));
		return true;
	}
}

template <typename T>
void readfile(const std::string & filename, 
				unsigned int & nx, unsigned int & ny, unsigned int & nz, unsigned int & nt,
				T & dx, T & dt,
				T & a_2, T & a_4, T & M, T & K, 
				unsigned int & t_skip)
{
	std::ifstream fin;
	std::string s;
	fin.open(filename.c_str());

    if (fin.fail())
    {
        std::cout << "Failed to open file " << filename << "..." << std::endl;
        exit(-1);
    }

	std::string key, value;
	std::cout << "Reading inputs from " << filename << "..." << std::endl;
	while (getline(fin, s))
	{
		if (!trim(s).empty() && parse_parameter(s, key, value))
		{
			if (key == "dx")
			{
				dx = (T)std::stod(value);
			}
			else if (key == "dt")
			{
				dt = (T)std::stod(value);
			}
			else if (key == "nx")
			{
				nx = (unsigned int)std::stoul(value);
			}
			else if (key == "ny")
			{
				ny = (unsigned int)std::stoul(value);
			}
            else if (key == "nz")
            {
                nz = (unsigned int)std::stoul(value);
            }
			else if (key == "nt")
			{
				nt = (unsigned int)std::stoul(value);
			}
			else if (key == "a_2")
			{
				a_2 = (T)std::stod(value);
			}
			else if (key == "a_4")
			{
				a_4 = (T)std::stod(value);
			}
			else if (key == "M")
			{
				M = (T)std::stod(value);
			}
			else if (key == "K")
			{
				K = (T)std::stod(value);
			}
			else if (key == "t_skip")
			{
				t_skip = (unsigned int)std::stoul(value);
			}
			else std::cout << key << " = " << value << " not understood!" << std::endl;
		}	
	}
	std::cout << "Done!" << std::endl;

	// Print inputs read in
	std::cout << "------- Inputs -------" << std::endl;
	std::cout << "nx = " << nx << std::endl;
	std::cout << "ny = " << ny << std::endl;
    std::cout << "nz = " << nz << std::endl;
	std::cout << "nt = " << nt << std::endl;
	std::cout << "dx = " << dx << std::endl;
	std::cout << "dt = " << dt << std::endl;
	std::cout << "a_2 = " << a_2 << std::endl;
	std::cout << "a_4 = " << a_4 << std::endl;
	std::cout << "M = " << M << std::endl;
	std::cout << "K = " << K << std::endl;
	std::cout << "t_skip = " << t_skip << std::endl;
	std::cout << "------- End -------" << std::endl;
}

template void readfile<double>(const std::string & filename, 
                unsigned int & nx, unsigned int & ny, unsigned int & nz, unsigned int & nt,
                double & dx, double & dt,
                double & a_2, double & a_4, double & M, double & K, 
                unsigned int & t_skip);

template void readfile<float>(const std::string & filename, 
                unsigned int & nx, unsigned int & ny, unsigned int & nz, unsigned int & nt, 
                float & dx, float & dt,
                float & a_2, float & a_4, float & M, float & K, 
                unsigned int & t_skip);



