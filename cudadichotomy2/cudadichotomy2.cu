// https://ru.wikipedia.org/wiki/���������

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/version.h>
#include <thrust/sort.h>
#include <cstdlib>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <clocale>
#include <functional> 
#include <cctype>
#include <locale>
#include <assert.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
using namespace thrust;

// Thrust is a C++ template library for CUDA based on the Standard Template Library (STL).
// Thrust allows you to implement high performance parallel applications with minimal programming effort through a high-level interface that is fully interoperable with CUDA C.
// Thrust provides a rich collection of data parallel primitives such as scan, sort, and reduce, which can be composed together to implement complex algorithms with concise, readable source code.
// By describing your computation in terms of these high-level abstractions you provide Thrust with the freedom to select the most efficient implementation automatically.
// As a result, Thrust can be utilized in rapid prototyping of CUDA applications, where programmer productivity matters most, as well as in production, where robustness and absolute performance are crucial.
// Read more at: http://docs.nvidia.com/cuda/thrust/index.html#ixzz3hymTnQwX 

double delta(thrust::device_vector<double> x,thrust::device_vector<double> y);
unsigned long total_of(thrust::device_vector<unsigned> m);

template <typename T>
struct inc_functor { 
	__host__ __device__ T operator()(const T& value) const 
	{
		return value+1; 
	} 
};

template <typename T>
struct square_functor { 
	__host__ __device__ T operator()(const T& value) const 
	{
		return value*value; 
	} 
};

template <typename T>
struct add_functor { 
	__host__ __device__ T operator()(const T& value1, const T& value2) const 
	{
		return value1+value2; 
	} 
};

template <typename T>
struct sub_functor { 
	__host__ __device__ T operator()(const T& value1, const T& value2) const 
	{
		return value1-value2; 
	} 
};

template <typename T>
struct mul_functor { 
	__host__ __device__ T operator()(const T& value1, const T& value2) const 
	{
		return value1*value2; 
	} 
};

template <typename T>
struct diff_functor { 
	__host__ __device__ T operator()(const T& value1, const T& value2) const 
	{
		return thrust::max(value1-value2,value2-value1); 
	} 
};

template <typename T>
struct abs_functor { 
	__host__ __device__ T operator()(const T& value) const 
	{
		return thrust::max(value,-value); 
	} 
};

template <typename T>
struct max_functor { 
	__host__ __device__ T operator()(const T& value1, const T& value2) const 
	{
		return thrust::max(value1,value2); 
	} 
};

/////////////////////////////////////////////////////////
// ���������� ����� ����� �������
// m - ����� ��������� �� ������� �� ���������
unsigned long total_of(thrust::device_vector<unsigned> m)
{
	return thrust::transform_reduce(m.begin(), m.end(), inc_functor<unsigned>(), 1UL, mul_functor<unsigned long>());
}

/////////////////////////////////////////////////////////
// ���������� ��������� ����� ����� ��������� ���������
double delta(thrust::device_vector<double> x,thrust::device_vector<double> y)
{
	unsigned i=thrust::min(x.size(),y.size());
	thrust::device_vector<double> diff(thrust::max(x.size(),y.size()));
	thrust::transform(x.begin(), x.begin()+i, y.begin(), diff.begin(), diff_functor<double>());
	thrust::transform(x.begin()+i, x.end(), diff.begin()+i, abs_functor<double>());
	thrust::transform(y.begin()+i, y.end(), diff.begin()+i, abs_functor<double>());
	return thrust::reduce(diff.begin(), diff.end(), 0.0, max_functor<double>());
}

/////////////////////////////////////////////////////////
// ����� ����������� �������
__device__ bool f1(double * x, int n)
{
	const double _c[] = {1, 1};
	const double b = 16;

	double y = 0.0;
	for(int i=0; i<n; i++) y+=(x[i]-_c[i])*(x[i]-_c[i]);
	return y<b;
}
__device__ bool f2(double * x, int n)
{
	const double _c[] = {2, 2};
	const double b = 16;

	double y = 0.0;
	for(int i=0; i<n; i++) y+=(x[i]-_c[i])*(x[i]-_c[i]);
	return y<b;
}
// ...

/////////////////////////////////////////////////////////
// ������� �������
__device__ double w(double * x, int n)
{
	const double _c[] = {2, 3};

	double y = 0.0;
	for(int i=0; i<n; i++) y+=(x[i]-_c[i])*(x[i]-_c[i]);
	return y;
}

enum t_ask_mode {
	NOASK = 0,
	ASK = 1
};
enum t_trace_mode {
	NOTRACE = 0,
	TRACE = 1
};
t_ask_mode ask_mode = NOASK;
t_trace_mode trace_mode = TRACE;

/////////////////////////////////////////////////////////
// ��������� ��������
const unsigned _n = 2;
const unsigned _m[] = {3, 3};
const double _a[] = {0, 0};
const double _b[] = {100, 100};
const double _e=1e-15;

/////////////////////////////////////////////////////////
// ���������� ������� �������� ��������� ������� �� ������ ����
// index - ����� ���� �������
__device__ void vector_of(unsigned * vector, unsigned long index, unsigned * m, int n)
{
	for(unsigned i=0;i<n;i++)
	{
		vector[i]=index%(1ul+m[i]);
		index/=1ul+m[i];
	}
}

/////////////////////////////////////////////////////////
// �������������� ������� �������� ��������� �������
// � ������ ��������� �����
// vector - ������ �������� ��������� �������
// m - ����� ��������� �� ������� �� ���������
// a - ������ ����������� ��������� �����
// b - ������ ������������ ��������� �����
__device__ void point_of(double * point, unsigned * vector,
						 unsigned * m, double * a, double * b, int n)
{
	for(unsigned i=0;i<n;i++) point[i]=(a[i]*(m[i]-vector[i])+b[i]*vector[i])/m[i];
}

/////////////////////////////////////////////////////////
// �������� �������������� ����� �������, �������� �������������
// x - ���������� �����
// f - ����� ����������� �������
__device__ bool check(double * x, int n)
{
	return f1(x,n)&&f2(x,n);
}

__device__ void copy(double * x, double * y, int n)
{
	for(int i=0; i<n; i++) x[i] = y[i];
}

__global__ void kernel(
	unsigned * vPtr,
	double * tPtr,
	double * xPtr,
	double * yPtr,
	bool * ePtr,
	unsigned * m,
	double * a,
	double * b,
	unsigned long total,
	int n)
{
	// �������� ������������� ����
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	unsigned * v = &vPtr[id*n];
	double * t = &tPtr[id*n];
	double * x = &xPtr[id*n];
	double * y = &yPtr[id];
	bool * e = &ePtr[id];

	*e = false;
	for (unsigned long i = blockDim.x*blockIdx.x + threadIdx.x;
		i < total;
		i += blockDim.x*gridDim.x) {
			vector_of(v, i, m, n);
			point_of(t, v, m, a, b, n);
			if(!check(t, n)) continue;
			if(!*e) {
				*y = w(t, n);
				copy(x, t, n);
				*e = true;
				continue;
			}
			double y1 =  w(t, n);
			if(y1 < *y) {
				copy(x, t, n);
				*y = y1;
			}
	}
}

int main(int argc, char* argv[])
{
  // http://stackoverflow.com/questions/2236197/what-is-the-easiest-way-to-initialize-a-stdvector-with-hardcoded-elements

	unsigned n=_n;
	double e=_e;
	thrust::host_vector<unsigned> hm(_m, _m + sizeof(_m) / sizeof(_m[0]) );
	thrust::host_vector<double> ha(_a, _a + sizeof(_a) / sizeof(_a[0]) );
	thrust::host_vector<double> hb(_b, _b + sizeof(_b) / sizeof(_b[0]) );

	char * input_file_name = NULL;
	char * output_file_name = NULL;

	int gridSize = 0;
	int blockSize = 0;

	// ��������� ��������� � ������� Windows
	// ������� setlocale() ����� ��� ���������, ������ �������� - ��� ��������� ������, � ����� ������ LC_TYPE - ����� ��������, ������ �������� � �������� ������. 
	// ������ ������� ��������� ����� ������ "Russian", ��� ��������� ������ ������� �������, ����� ����� �������� ����� ����� �� ��� � � ��.
	setlocale(LC_ALL, "");

	for(int i=1; i<argc; i++)
	{
		if(strcmp(argv[i],"-help")==0) 
		{
			std::cout << "�������� ������� �������� ��������� �������" << std::endl;
			//			std::cout << "\t-n <����������� ������������>" << std::endl;
			std::cout << "\t-m <����� ��������� �� ������� �� ���������>" << std::endl;
			std::cout << "\t-a <����������� ���������� �� ������� �� ���������>" << std::endl;
			std::cout << "\t-b <������������ ���������� �� ������� �� ���������>" << std::endl;
			std::cout << "\t-e <������� ����������>" << std::endl;
			std::cout << "\t-ask/noask" << std::endl;
			std::cout << "\t-trace/notrace" << std::endl;
			std::cout << "\t��. https://ru.wikipedia.org/wiki/���������" << std::endl;
		}
		else if(strcmp(argv[i],"-ask")==0) ask_mode = ASK;
		else if(strcmp(argv[i],"-noask")==0) ask_mode = NOASK;
		else if(strcmp(argv[i],"-trace")==0) trace_mode = TRACE;
		else if(strcmp(argv[i],"-notrace")==0) trace_mode = NOTRACE;
		//		else if(strcmp(argv[i],"-n")==0) n = atoi(argv[++i]);
		else if(strcmp(argv[i],"-e")==0) e = atof(argv[++i]);
		else if(strcmp(argv[i],"-m")==0) {
			std::istringstream ss(argv[++i]);
			hm.clear();
			for(unsigned i=0;i<n;i++) hm.push_back(atoi(argv[++i]));
		}
		else if(strcmp(argv[i],"-a")==0) {
			ha.clear();
			for(unsigned i=0;i<n;i++) ha.push_back(atof(argv[++i]));
		}
		else if(strcmp(argv[i],"-b")==0) {
			hb.clear();
			for(unsigned i=0;i<n;i++) hb.push_back(atof(argv[++i]));
		}
		else if(strcmp(argv[i],"-input")==0) input_file_name = argv[++i];
		else if(strcmp(argv[i],"-output")==0) output_file_name = argv[++i];
		else if(strcmp(argv[i],"g")==0) gridSize = atoi(argv[++i]);
		else if(strcmp(argv[i],"b")==0) blockSize = atoi(argv[++i]);
	}

	if(input_file_name!=NULL) freopen(input_file_name,"r",stdin);
	if(output_file_name!=NULL) freopen(output_file_name,"w",stdout);


	if(ask_mode == ASK)
	{
		//  std::cout << "������� ����������� ������������:"<< std::endl; std::cin >> n;

		std::cout << "������� ����� ��������� �� ������� �� ��������� m[" << n << "]:"<< std::endl;
		hm.clear();
		for(unsigned i=0;i<n;i++)
		{
			int x; 
			std::cin >> x;
			hm.push_back(x);
		}

		std::cout << "������� ����������� ���������� �� ������� �� ��������� a[" << n << "]:"<< std::endl;
		ha.clear();
		for(unsigned i=0;i<n;i++)
		{
			double x; 
			std::cin >> x;
			ha.push_back(x);
		}

		std::cout << "������� ������������ ���������� �� ������� �� ��������� b[" << n << "]:"<< std::endl;
		hb.clear();
		for(unsigned i=0;i<n;i++)
		{
			double x; 
			std::cin >> x;
			hb.push_back(x);
		}

		std::cout << "������� �������� ����������:"<< std::endl; std::cin >> e;
	}

	// Find/set the device.
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	for (int i = 0; i < device_count; ++i)
	{
		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, i);
		std::cout << "Running on GPU " << i << " (" << properties.name << ")" << std::endl;
	}

	int major = THRUST_MAJOR_VERSION;
	int minor = THRUST_MINOR_VERSION;

	std::cout << "Thrust v" << major << "." << minor << std::endl;
	std::cout << "����������� ������������ : " << n << std::endl;
	std::cout << "����� ���������          : "; for(unsigned i=0;i<hm.size();i++) std::cout << hm[i] << " "; std::cout << std::endl; 
	std::cout << "����������� ����������   : "; for(unsigned i=0;i<ha.size();i++) std::cout << ha[i] << " "; std::cout << std::endl; 
	std::cout << "������������ ����������  : "; for(unsigned i=0;i<hb.size();i++) std::cout << hb[i] << " "; std::cout << std::endl; 
	std::cout << "�������� ����������      : " << e << std::endl;

	thrust::device_vector<unsigned> m(hm);
	thrust::device_vector<double> a(ha);
	thrust::device_vector<double> b(hb);

	for(unsigned i=0;i<m.size();i++) assert(m[i]>2);

// ��������
	unsigned long total=total_of(m);

	// ��������� ����������� ��������� �� ��������, ����

	int blocks = (gridSize > 0)? gridSize : thrust::min(15, (int)pow(total,0.333333));
	int threads = (blockSize > 0)? blockSize : thrust::min(15, (int)pow(total,0.333333));

	// ���������� ������ ��� ������������ ����������
	thrust::device_vector<unsigned> vArray(blocks*threads*n);
	thrust::device_vector<double> tArray(blocks*threads*n);
	thrust::device_vector<double> xArray(blocks*threads*n);
	thrust::device_vector<double> yArray(blocks*threads);
	thrust::device_vector<bool> eArray(blocks*threads);

	unsigned * vPtr = thrust::raw_pointer_cast(&vArray[0]);
	double * tPtr = thrust::raw_pointer_cast(&tArray[0]);
	double * xPtr = thrust::raw_pointer_cast(&xArray[0]);
	double * yPtr = thrust::raw_pointer_cast(&yArray[0]);
	bool * ePtr = thrust::raw_pointer_cast(&eArray[0]);
	unsigned * mPtr = thrust::raw_pointer_cast(&m[0]);
	double * aPtr = thrust::raw_pointer_cast(&a[0]);
	double * bPtr = thrust::raw_pointer_cast(&b[0]);

	thrust::device_vector<double> x(n);
	double y;

	while(true)
	{
		kernel<<< blocks, threads >>>(vPtr, tPtr, xPtr, yPtr, ePtr, mPtr, aPtr, bPtr, total, n);

		thrust::host_vector<double> hyArray(yArray);

		// ������� ������ ����� � �������, �������� �������������
		auto it = thrust::find(eArray.begin(), eArray.end(), true);
		int index = thrust::distance(eArray.begin(), it);
		y=hyArray[index];
		thrust::copy(&xArray[index*n], &xArray[index*n+n], x.begin());

		while((it = thrust::find(it, eArray.end(), true)) < eArray.end())
		{
			int index = thrust::distance(eArray.begin(), it++);
			if(y<hyArray[index]) continue;
			y=hyArray[index];
			thrust::copy(&xArray[index*n], &xArray[index*n+n], x.begin());
		}

		if(trace_mode==TRACE) {
			thrust::host_vector<double> hx(x);
			for(unsigned i=0;i<hx.size();i++) std::cout << hx[i] << " "; 
		}
		if(trace_mode==TRACE) std::cout << "-> " << y << std::endl; 

		if(delta(a,b)<e) break;

		for(unsigned i=0;i<thrust::min(a.size(),b.size());i++) {
			double aa = a[i];
			double bb = b[i];
			double xx = x[i];
			a[i]=thrust::max(aa,xx-(bb-aa)/m[i]);
			b[i]=thrust::min(bb,xx+(bb-aa)/m[i]);
		}
	}

	thrust::host_vector<double> hx(x);
	std::cout << "����� ��������           : "; for(unsigned i=0;i<hx.size();i++) std::cout << hx[i] << " "; std::cout << std::endl; 
	std::cout << "����������� ��������     : " << y << std::endl; 
	std::cout << "��. https://ru.wikipedia.org/wiki/���������" << std::endl;

	getchar();
	getchar();

	return 0;
}
