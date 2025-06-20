#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cuComplex.h>
#include <cuda_runtime.h>

#include "cudss.h"
#include "MathUtils.h"
#include "defines.h"
#include "Utils.h"

using namespace Utils;

#define IS_FULL_MATRIX 0
#define HYBRID_MEMORY_MODE 0

int main(){
	try{
		logGPUMemoryUsage("Init");
		cudaError_t cuda_error = cudaSuccess;
		cudssStatus_t status = CUDSS_STATUS_SUCCESS;
		// CSR形式のデータをファイルから読み込む
		// std::string file_path = "./matrix_data/1015/";
		// std::string file_path = "./matrix_data/14557/";
		// std::string file_path = "/home/goto/data/SparseMatrix/99403/";
		// std::string file_path = "/home/goto/data/SparseMatrix/607232/";
		std::string file_path = "/home/goto/data/SparseMatrix/1M/";
		// std::string file_path = "/home/goto/data/SparseMatrix/14557/";
		// std::string file_path = "/home/goto/data/SparseMatrix/1015/";
		// std::string file_path = "/home/goto/data/matrix_data/1M/";
		// std::string file_path = "/home/goto/data/matrix_data/2.5M/";
		// std::string file_path = "/home/goto/data/matrix_data/1000/";
		// std::string file_path = "/home/goto/data/matrix_data/1M_2/";
		// std::string file_path = "/home/goto/data/matrix_data/173294/";
		// std::string file_path = "/home/goto/data/SparseMatrix/others/82158/";
		// std::string file_path = "./matrix_data/2/";
		// std::string file_path = "./";
		std::vector<int> _row_ptr = loadData<int>(file_path + "row_ptr.dat");
		std::vector<int> _col_idx = loadData<int>(file_path + "col_idx.dat");
		std::vector<std::complex<double>> _values_tmp = loadData<std::complex<double>>(file_path + "values.dat");
		std::vector<cuDoubleComplex> _values(_values_tmp.size());
		for(size_t ii=0; ii<_values.size(); ++ii){
			_values[ii] = make_cuDoubleComplex(_values_tmp[ii].real(), _values_tmp[ii].imag());
		}
		MathUtils::CSR<cuDoubleComplex> org_mat{_row_ptr, _col_idx, _values};
#if IS_FULL_MATRIX == 1
		std::cout << "FULL" << std::endl;
		const MathUtils::CSR<cuDoubleComplex> full_mat{_row_ptr, _col_idx, _values};
#elif IS_FULL_MATRIX == 0
		std::cout << "UPPER" << std::endl;
		const auto full_mat = MathUtils::transformFullMatrix(org_mat);
#endif
		std::vector<int> row_ptr = full_mat.row_ptr;
		std::vector<int> col_idx = full_mat.col_idx;
		std::vector<cuDoubleComplex> values = full_mat.values;

		// 行列のサイズと非ゼロ要素数の推定
		int n = row_ptr.size() - 1;
		int nnz = values.size();
		int nrhs = 1; // -> ここでは右辺ベクトルの数は1に固定
		std::cout << "row_ptr size: " << row_ptr.size() << std::endl;
		std::cout << "n: " << n << std::endl;
		std::cout << "nnz: " << nnz << std::endl;

		// 右辺ベクトルbの初期化（例として1+0iのベクトルを使用）
		std::vector<cuDoubleComplex> b_values(n, cuDoubleComplex{1.0, 0.0});
		std::vector<cuDoubleComplex> x_values(n);

		// device側でのcsrデータの作成
		int *row_ptr_d = nullptr;
		int *col_idx_d = nullptr;
		cuDoubleComplex *values_d = nullptr;
		cuDoubleComplex *b_values_d = nullptr;
		cuDoubleComplex *x_values_d = nullptr;

		logGPUMemoryUsage("Before CUDA malloc");

		CUDA_CALL_AND_CHECK(cudaMalloc(&row_ptr_d, (n+1)*sizeof(int)), "cudaMalloc success");
		CUDA_CALL_AND_CHECK(cudaMalloc(&col_idx_d, nnz * sizeof(int)), "cudaMalloc success");
		CUDA_CALL_AND_CHECK(cudaMalloc(&values_d,  nnz * sizeof(cuDoubleComplex)), "cudaMalloc success");
		CUDA_CALL_AND_CHECK(cudaMalloc(&b_values_d,       n * sizeof(cuDoubleComplex)), "cudaMalloc success");
		CUDA_CALL_AND_CHECK(cudaMalloc(&x_values_d,       n * sizeof(cuDoubleComplex)), "cudaMalloc success");

		cudaMemcpy(row_ptr_d, row_ptr.data(), (n+1) * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(col_idx_d, col_idx.data(), nnz   * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(values_d, values.data(),   nnz   * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
		cudaMemcpy(b_values_d, b_values.data(), n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

		logGPUMemoryUsage("After CUDA malloc");

		// Create a CUDA stream
		cudaStream_t stream = nullptr;
		cudaStreamCreate(&stream);

		// cuDSSハンドルetc.の初期化
		cudssHandle_t handle;
		cudssCreate(&handle);

		cudssSetStream(handle, stream);

		cudssConfig_t solverConfig;
		cudssConfigCreate(&solverConfig);

		cudssData_t solverData;
		cudssDataCreate(handle, &solverData);

		// Create matrix objects for the rhs b and solution x.
		cudssMatrix_t x, b;

		int64_t nrows = n, ncols = n;
		int ldb = ncols, ldx = nrows;

		cudssMatrixCreateDn(&b, ncols, nrhs, ldb, b_values_d, CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR);
		cudssMatrixCreateDn(&x, nrows, nrhs, ldx, x_values_d, CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR);

		// Create a matrix object for the sparse input matrix A.
		cudssMatrix_t A;
		cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;
		// cudssMatrixType_t mtype = CUDSS_MTYPE_SYMMETRIC;
		// cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
		cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;
		cudssIndexBase_t base = CUDSS_BASE_ZERO;

		cudssMatrixCreateCsr(&A, nrows, ncols, nnz, row_ptr_d, nullptr, col_idx_d, values_d, CUDA_R_32I, CUDA_C_64F, mtype, mview, base);

#if HYBRID_MEMORY_MODE
		// Hybrid memory modeをオンにする
		int hybrid_mode = 1;
		cudssConfigSet(solverConfig, CUDSS_CONFIG_HYBRID_MODE, &hybrid_mode, sizeof(hybrid_mode));
#endif

		// 時間計測をするためにイベントを作成
		cudaEvent_t start_analysis, stop_analysis;
		cudaEvent_t start_factor, stop_factor;
		cudaEvent_t start_solve, stop_solve;
		cudaEventCreate(&start_analysis);
		cudaEventCreate(&stop_analysis);
		cudaEventCreate(&start_factor);
		cudaEventCreate(&stop_factor);
		cudaEventCreate(&start_solve);
		cudaEventCreate(&stop_solve);

		float analysis_time_ms = 0.0f;
		float factor_time_ms = 0.0f;
		float solve_time_ms = 0.0f;
		float total_time_ms = 0.0f;


		// Symbolic factorization
		logGPUMemoryUsage("Before reorder");
		cudaEventRecord(start_analysis, stream);
		CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData, A, x, b), status, "cudssExecute");
		cudaEventRecord(stop_analysis, stream);
		cudaEventSynchronize(stop_analysis);
		cudaEventElapsedTime(&analysis_time_ms, start_analysis, stop_analysis);
		std::cout << "Symbolic factorization completed. Time = " << analysis_time_ms << " [ms]" << std::endl;
		logGPUMemoryUsage("After reorder");


#if HYBRID_MEMORY_MODE
		size_t size_written;
		int64_t device_memory_min;
		cudssDataGet(handle, solverData, CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN, &device_memory_min, sizeof(device_memory_min), &size_written); 
		std::cout << "hybrid memory mode: " << device_memory_min << " bytes" << std::endl;
		// int64_t hybrid_device_memory_limit = 1.6*10e9;
		// cudssConfigSet(solverConfig, CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT, &hybrid_device_memory_limit, sizeof(hybrid_device_memory_limit));
		// std::cout << "hybrid memory mode limit: " << hybrid_device_memory_limit << " bytes" << std::endl;
#endif

		// Factorization
		logGPUMemoryUsage("Before factorization");
		cudaEventRecord(start_factor, stream);
		CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig, solverData, A, x, b), status, "cudssExecute");
		cudaEventRecord(stop_factor, stream);
		cudaEventSynchronize(stop_factor);
		cudaEventElapsedTime(&factor_time_ms, start_factor, stop_factor);
		std::cout << "Factorization completed. Time = " << factor_time_ms << " [ms]" << std::endl;
		logGPUMemoryUsage("After factorization");

		// int64_t lu_nnz_num;
		// cudssDataGet(handle, solverData, CUDSS_DATA_LU_NNZ, &lu_nnz_num, sizeof(lu_nnz_num), &size_written); 
		// std::cout << "lu nnz num: " << lu_nnz_num << " bytes" << std::endl;

		// Solving
		logGPUMemoryUsage("Before solve");
		cudaEventRecord(start_solve, stream);
		CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData, A, x, b), status, "cudssExecute");
		cudaEventRecord(stop_solve, stream);
		cudaEventSynchronize(stop_solve);
		cudaEventElapsedTime(&solve_time_ms, start_solve, stop_solve);
		std::cout << "Solving completed. Time = " << solve_time_ms << " [ms]" << std::endl;
		logGPUMemoryUsage("After solve");

		cudaEventElapsedTime(&total_time_ms, start_analysis, stop_solve);
		std::cout << "DSS completed. Total time = " << total_time_ms << " [ms]" << std::endl;

		// Destroying opaque objects, matrix wrappers and the cuDSS library handle
		cudssMatrixDestroy(A);
		cudssMatrixDestroy(b);
		cudssMatrixDestroy(x);
		cudssDataDestroy(handle, solverData);
		cudssConfigDestroy(solverConfig);
		cudssDestroy(handle);
		logGPUMemoryUsage("After destroying matrix data");

		cudaStreamSynchronize(stream);

		// 結果をホストにコピー
		cudaMemcpy(x_values.data(), x_values_d, nrhs * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
		std::ofstream out(file_path + "x.dat");
		for (int i = 0; i < n; i++) {
			out << "x[" << i << "] = (" << x_values[i].x << ", " << x_values[i].y << ")" << std::endl;
		}
		out.close();
		std::vector<cuDoubleComplex> b_confirm(n, cuDoubleComplex{0., 0.});
		for(size_t ii=0; ii<n; ++ii){
			for(size_t jj=row_ptr[ii]; jj<row_ptr[ii+1]; ++jj){
				const auto& col = col_idx[jj];
				const auto& value = values[jj];
				const auto& mul_ans = cuCmul(value, x_values[col]);
				b_confirm[ii].x += mul_ans.x;
				b_confirm[ii].y += mul_ans.y;
			}
		}
		out.open(file_path + "b.dat");
		for (int i = 0; i < n; i++) {
			out << "b[" << i << "] = (" << b_confirm[i].x << ", " << b_confirm[i].y << ")" << std::endl;
		}
		out.close();

		/// メモリ解放
		cudaFree(row_ptr_d);
		cudaFree(col_idx_d);
		cudaFree(values_d);
		cudaFree(b_values_d);
		cudaFree(x_values_d);

		logGPUMemoryUsage("After cudaFree data");
	}
	catch(const std::exception& e){
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}
	std::cout << "Success" << std::endl;
	return 0;
}
