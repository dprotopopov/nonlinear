// �������� ����������� ����������� � �������������� ������ �������

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

double module(thrust::device_vector<double> &x);
double delta(thrust::device_vector<double> &x, thrust::device_vector<double> &y);
unsigned long total_of(thrust::device_vector<size_t> &m);

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
unsigned long total_of(thrust::device_vector<size_t> &m)
{
	return thrust::transform_reduce(m.begin(), m.end(), inc_functor<size_t>(), 1UL, mul_functor<unsigned long>());
}

/////////////////////////////////////////////////////////
// ���������� ������ �������
double module(thrust::device_vector<double> &x)
{
	return thrust::transform_reduce(x.begin(), x.end(), abs_functor<double>(), 0.0, max_functor<double>());
}

/////////////////////////////////////////////////////////
// ���������� ��������� ����� ����� ��������� ���������
double delta(thrust::device_vector<double> &x, thrust::device_vector<double> &y)
{
	size_t i=thrust::min(x.size(),y.size());
	thrust::device_vector<double> diff(thrust::max(x.size(),y.size()));
	thrust::transform(x.begin(), x.begin()+i, y.begin(), diff.begin(), diff_functor<double>());
	thrust::transform(x.begin()+i, x.end(), diff.begin()+i, abs_functor<double>());
	thrust::transform(y.begin()+i, y.end(), diff.begin()+i, abs_functor<double>());
	return thrust::reduce(diff.begin(), diff.end(), 0.0, max_functor<double>());
}

/////////////////////////////////////////////////////////
// ����� ����������� �������
__device__ bool f1(double * x, size_t n)
{
	const double _c[] = {0, 0};
	const double b = 500;

	double y = 0.0;
	for(size_t i=0; i<n; i++) y+=(x[i]-_c[i])*(x[i]-_c[i]);
	return y<b*b;
}
__device__ bool f2(double * x, size_t n)
{
	const double _c[] = {100, 100};
	const double b = 500;

	double y = 0.0;
	for(size_t i=0; i<n; i++) y+=(x[i]-_c[i])*(x[i]-_c[i]);
	return y<b*b;
}
// ...

/////////////////////////////////////////////////////////
// ������� �������
__device__ double w(double * x, size_t n)
{
	const double _c0[] = {0, 0};
	const double _c1[] = {150, 180};
	const double _c2[] = {240, 200};
	const double _c3[] = {260, 90};
	const double _q[] = {3040, 1800, 800, 1200};
	double square[] = { 0, 0, 0, 0 };

	double y = 0.0;
	for(size_t i=0; i<n; i++) square[0]+=(x[i]-_c0[i])*(x[i]-_c0[i]);
	for(size_t i=0; i<n; i++) square[1]+=(x[i]-_c1[i])*(x[i]-_c1[i]);
	for(size_t i=0; i<n; i++) square[2]+=(x[i]-_c2[i])*(x[i]-_c2[i]);
	for(size_t i=0; i<n; i++) square[3]+=(x[i]-_c3[i])*(x[i]-_c3[i]);
	for(size_t i=0; i<4; i++) y+=_q[i]*sqrt(square[i]);
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
t_trace_mode trace_mode = NOTRACE;

/////////////////////////////////////////////////////////
// ��������� ��������
static const unsigned _count = 1;
static const size_t _n = 2;
static const size_t _m[] = {3, 3};
static const double _a[] = {0, 0};
static const double _b[] = {1000, 1000};
static const double _e=1e-8;

/////////////////////////////////////////////////////////
// ���������� ������� �������� ��������� ������� �� ������ ����
// index - ����� ���� �������
__device__ void vector_of(unsigned * vector, unsigned long index, size_t * m, size_t n)
{
	for(size_t i=0;i<n;i++)
	{
		unsigned long m1 = 1ul+m[i];
		vector[i]=index%m1;
		index/=m1;
	}
}

/////////////////////////////////////////////////////////
// �������������� ������� �������� ��������� �������
// � ������ ��������� �����
// vector - ������ �������� ��������� �������
// m - ����� ��������� �� ������� �� ���������
// a - ������ ����������� ��������� �����
// b - ������ ������������ ��������� �����
__device__ void point_of(double * point, unsigned * vector, size_t * m, double * a, double * b, size_t n)
{
	for(size_t i=0;i<n;i++) point[i]=(a[i]*(m[i]-vector[i])+b[i]*vector[i])/m[i];
}

/////////////////////////////////////////////////////////
// �������� �������������� ����� �������, �������� �������������
// x - ���������� �����
// f - ����� ����������� �������
// a - ������ ����������� ��������� �����
// b - ������ ������������ ��������� �����
__device__ bool check(double * x, double * a, double * b, size_t n)
{
	for(size_t i=0;i<n;i++) if(x[i]<a[i]&&x[i]<b[i]) return false;
	for(size_t i=0;i<n;i++) if(x[i]>a[i]&&x[i]>b[i]) return false;
	return f1(x,n)&&f2(x,n);
}

__device__ void copy(double * x, double * y, size_t n)
{
	for(size_t i=0; i<n; i++) x[i] = y[i];
}

__global__ void kernel(
	unsigned * vPtr,
	double * tPtr,
	double * xPtr,
	double * yPtr,
	bool * ePtr,
	size_t * m,
	double * a,
	double * b,
	unsigned long total,
	size_t n)
{
	// �������� ������������� ����
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	unsigned * v = &vPtr[id*n];
	double * t = &tPtr[id*n];
	double * x = &xPtr[id*n];
	double * y = &yPtr[id];
	bool * e = &ePtr[id];

	*e = false;
	for (unsigned long index = blockDim.x*blockIdx.x + threadIdx.x;
		index < total;
		index += blockDim.x*gridDim.x) {
			vector_of(v, index, m, n);
			point_of(t, v, m, a, b, n);
			if(!check(t, a, b, n)) continue;
			if(!*e) {
				::copy(x, t, n);
				*y = w(t, n);
				*e = true;
				continue;
			}
			double y1 =  w(t, n);
			if(y1 < *y) {
				::copy(x, t, n);
				*y = y1;
			}
	}
}

int main(int argc, char* argv[])
{
  // http://stackoverflow.com/questions/2236197/what-is-the-easiest-way-to-initialize-a-stdvector-with-hardcoded-elements

	unsigned count=_count;
	size_t n=_n;
	double e=_e;
	thrust::host_vector<size_t> hm(_m, _m + sizeof(_m) / sizeof(_m[0]) );
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
			std::cout << "Usage :\t" << argv[0] << " [...] [g <gridSize>] [b <blockSize>] [-input <inputfile>] [-output <outputfile>]" << std::endl;
			std::cout << "�������� ����������� ����������� � �������������� ������ �������" << std::endl;
			std::cout << "�������� ������� �������� ��������� �������" << std::endl;
			//			std::cout << "\t-n <����������� ������������>" << std::endl;
			std::cout << "\t-c <���������� ���������� ��������� ��� ������ �������>" << std::endl;
			std::cout << "\t-m <����� ��������� �� ������� �� ���������>" << std::endl;
			std::cout << "\t-a <����������� ���������� �� ������� �� ���������>" << std::endl;
			std::cout << "\t-b <������������ ���������� �� ������� �� ���������>" << std::endl;
			std::cout << "\t-e <�������� ����������>" << std::endl;
			std::cout << "\t-ask/noask" << std::endl;
			std::cout << "\t-trace/notrace" << std::endl;
		}
		else if(strcmp(argv[i],"-ask")==0) ask_mode = ASK;
		else if(strcmp(argv[i],"-noask")==0) ask_mode = NOASK;
		else if(strcmp(argv[i],"-trace")==0) trace_mode = TRACE;
		else if(strcmp(argv[i],"-notrace")==0) trace_mode = NOTRACE;
		//		else if(strcmp(argv[i],"-n")==0) n = atoi(argv[++i]);
		else if(strcmp(argv[i],"-e")==0) e = atof(argv[++i]);
		else if(strcmp(argv[i],"-c")==0) count = atoi(argv[++i]);
		else if(strcmp(argv[i],"-m")==0) {
			std::istringstream ss(argv[++i]);
			hm.clear();
			for(size_t i=0;i<n;i++) hm.push_back(atoi(argv[++i]));
		}
		else if(strcmp(argv[i],"-a")==0) {
			ha.clear();
			for(size_t i=0;i<n;i++) ha.push_back(atof(argv[++i]));
		}
		else if(strcmp(argv[i],"-b")==0) {
			hb.clear();
			for(size_t i=0;i<n;i++) hb.push_back(atof(argv[++i]));
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
		for(size_t i=0;i<n;i++)
		{
			int x; 
			std::cin >> x;
			hm.push_back(x);
		}

		std::cout << "������� ����������� ���������� �� ������� �� ��������� a[" << n << "]:"<< std::endl;
		ha.clear();
		for(size_t i=0;i<n;i++)
		{
			double x; 
			std::cin >> x;
			ha.push_back(x);
		}

		std::cout << "������� ������������ ���������� �� ������� �� ��������� b[" << n << "]:"<< std::endl;
		hb.clear();
		for(size_t i=0;i<n;i++)
		{
			double x; 
			std::cin >> x;
			hb.push_back(x);
		}

		std::cout << "������� �������� ����������:"<< std::endl; std::cin >> e;
		std::cout << "������� ���������� ���������� ��������� ��� ������ �������:"<< std::endl; std::cin >> count;
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

	for(size_t i=0;i<hm.size();i++) assert(hm[i]>2);

// ��������
	clock_t time = clock();

	thrust::device_vector<size_t> m(hm);
	thrust::device_vector<double> a(ha);
	thrust::device_vector<double> b(hb);

	unsigned long total=total_of(m);

	// ��������� ����������� ��������� �� ��������, ����

	int blocks = (gridSize > 0)? gridSize : thrust::min(15, (int)pow(total,0.333333));
	int threads = (blockSize > 0)? blockSize : thrust::min(15, (int)pow(total,0.333333));

	// ���������� ������ ��� ������������ ����������
	thrust::host_vector<double> hyArray(blocks*threads);
	thrust::device_vector<unsigned> vArray(blocks*threads*n);
	thrust::device_vector<double> tArray(blocks*threads*n);
	thrust::device_vector<double> xArray(blocks*threads*n);
	thrust::device_vector<double> yArray(blocks*threads);
	thrust::device_vector<bool> eArray(blocks*threads);
	thrust::device_vector<double> a1(n);
	thrust::device_vector<double> b1(n);

	unsigned * vPtr = thrust::raw_pointer_cast(&vArray[0]);
	double * tPtr = thrust::raw_pointer_cast(&tArray[0]);
	double * xPtr = thrust::raw_pointer_cast(&xArray[0]);
	double * yPtr = thrust::raw_pointer_cast(&yArray[0]);
	bool * ePtr = thrust::raw_pointer_cast(&eArray[0]);
	size_t * mPtr = thrust::raw_pointer_cast(&m[0]);
	double * aPtr = thrust::raw_pointer_cast(&a[0]);
	double * bPtr = thrust::raw_pointer_cast(&b[0]);
	double * aPtr1 = thrust::raw_pointer_cast(&a1[0]);
	double * bPtr1 = thrust::raw_pointer_cast(&b1[0]);

	thrust::host_vector<double> hx(n);

	// ��������

	thrust::device_vector<double> x(n);
	double y;

	if(trace_mode==TRACE&&count==1) std::cout << "for #1" << std::endl; 
	for(unsigned s=0; s<count; s++)
	{
		thrust::copy(a.begin(), a.end(), a1.begin());
		thrust::copy(b.begin(), b.end(), b1.begin());

		if(trace_mode==TRACE&&count==1) std::cout << "while #1" << std::endl; 
		while(true)
		{
			// ������� ������ ����� � �������, �������� �������������
			unsigned long total=total_of(m);

			if(trace_mode==TRACE&&count==1) std::cout << "kernel" << std::endl; 
			kernel<<< blocks, threads >>>(vPtr, tPtr, xPtr, yPtr, ePtr, mPtr, aPtr1, bPtr1, total, n);
			thrust::copy(yArray.begin(), yArray.end(), hyArray.begin());

			auto it = thrust::find(eArray.begin(), eArray.end(), true);
			if(it>=eArray.end())
			{
				for(size_t i=0; i<n; i++) m[i]<<=1u;
				continue;
			}

			size_t index = thrust::distance(eArray.begin(), it++);
			y=hyArray[index];
			thrust::copy(&xArray[index*n], &xArray[index*n+n], x.begin());

			if(trace_mode==TRACE&&count==1) {
				thrust::copy(x.begin(), x.end(), hx.begin());
				for(size_t i=0;i<hx.size();i++) std::cout << hx[i] << " "; 
			}
			if(trace_mode==TRACE&&count==1) std::cout << "-> " << y << std::endl; 

			while((it = thrust::find(it, eArray.end(), true)) < eArray.end())
			{
				size_t index = thrust::distance(eArray.begin(), it++);
				double y1=hyArray[index];
				if(y<y1) continue;
				y=y1;
				thrust::copy(&xArray[index*n], &xArray[index*n+n], x.begin());

				if(trace_mode==TRACE&&count==1) {
					thrust::copy(x.begin(), x.end(), hx.begin());
					for(size_t i=0;i<hx.size();i++) std::cout << hx[i] << " "; 
				}
				if(trace_mode==TRACE&&count==1) std::cout << "-> " << y << std::endl; 
			}

			double dd = delta(a1,b1);
			double cc = thrust::max(module(a1),module(b1));
			if(dd<=cc*e) break;

			for(size_t k=0;k<n;k++) {
				double ak = a1[k];
				double bk = b1[k];
				double xk = x[k];
				double dd = thrust::max(ak-bk,bk-ak);
				a1[k]=thrust::max(ak,xk-dd/m[k]);
				b1[k]=thrust::min(bk,xk+dd/m[k]);
			}
		}
	}

	time = clock() - time;
	double seconds = ((double)time)/CLOCKS_PER_SEC/count;

	thrust::copy(x.begin(), x.end(), hx.begin());
	std::cout << "����������� ����         : " << argv[0] << std::endl;
	std::cout << "����������� ������������ : " << n << std::endl;
	std::cout << "����� ���������          : "; for(size_t i=0;i<hm.size();i++) std::cout << hm[i] << " "; std::cout << std::endl; 
	std::cout << "����������� ����������   : "; for(size_t i=0;i<ha.size();i++) std::cout << ha[i] << " "; std::cout << std::endl; 
	std::cout << "������������ ����������  : "; for(size_t i=0;i<hb.size();i++) std::cout << hb[i] << " "; std::cout << std::endl; 
	std::cout << "�������� ����������      : " << e << std::endl;
	std::cout << "����� ��������           : "; for(size_t i=0;i<hx.size();i++) std::cout << hx[i] << " "; std::cout << std::endl; 
	std::cout << "����������� ��������     : " << y << std::endl; 
	std::cout << "����� ���������� (���.)  : " << seconds << std::endl; 

	getchar();
	getchar();

	return 0;
}

