#include <vector>
#include <string>
#include <fstream>

namespace Utils{

template <class T>
std::vector<T> loadData(const std::string& filename){
    std::ifstream file(filename);
    if(!file){
        throw std::runtime_error("cannot open this file: " + filename);
    }
    std::vector<T> data;
    T value;
    while(file >> value){
        data.push_back(value);
    }
    return data;
}

// std::complex<double>型用の特殊化テンプレート
template <>
std::vector<std::complex<double>> loadData<std::complex<double>>(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("ファイルを開けません: " + filename);
    }
    std::vector<std::complex<double>> data;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double real, imag;
        if (!(iss >> real >> imag)) {
            throw std::runtime_error("データ形式が不正です: " + line);
        }
        data.emplace_back(real, imag);
    }
    return data;
}

template<class T>
void printVector(const std::vector<T>& data){
    for(auto&& e: data){
        std::cout << e << std::endl;
    }
}

template<>
void printVector<std::complex<double>>(const std::vector<std::complex<double>>& data){
    for(auto&& e: data){
        std::cout << e.real() << ", " << e.imag() << std::endl;
    }
}

void logGPUMemoryUsage(const std::string& stage) {
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	std::cout << "[" << stage << "] GPU Memory Usage: "
		// << "Free = " << free_mem / (1024.0 * 1024.0) << " MB, "
		// << "Total = " << total_mem / (1024.0 * 1024.0) << " MB, "
		<< "Used = " << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB" 
		<< std::endl;
}


} // namespace Utils
