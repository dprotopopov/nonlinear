// Алгоритм Хука и Дживса с использованием одномерной минимизации
// Базара М., Шетти К.
// Нелинейное программирование. Теория и алгоритмы:
// Пер. с англ. - М.: Мир, 1982.
// 583 с.

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
double distance(thrust::device_vector<double> x,thrust::device_vector<double> y);
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
// Вычисление числа узлов решётки
// m - число сегментов по каждому из измерений
unsigned long total_of(thrust::device_vector<unsigned> m)
{
	return thrust::transform_reduce(m.begin(), m.end(), inc_functor<unsigned>(), 1UL, mul_functor<unsigned long>());
}

/////////////////////////////////////////////////////////
// Вычисление растояния между двумя векторами координат
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
// Вычисление растояния между двумя векторами координат
double distance(thrust::device_vector<double> x,thrust::device_vector<double> y)
{
	unsigned i=thrust::min(x.size(),y.size());
	thrust::device_vector<double> sub(std::max(x.size(),y.size()));
	thrust::transform(x.begin(), x.end(), y.begin(), sub.begin(), sub_functor<double>());
	thrust::copy(x.begin()+i, x.end(), sub.begin()+i);
	thrust::copy(y.begin()+i, y.end(), sub.begin()+i);
	return std::sqrt(thrust::transform_reduce(sub.begin(), sub.end(), square_functor<double>(), 0.0, add_functor<double>()));
}

/////////////////////////////////////////////////////////
// Набор проверочных функций
__device__ bool f1(double * x, int n)
{
	const double _c[] = {1, 1};
	const double b = 16;

	double y = 0.0;
	for(unsigned i=0; i<n; i++) y+=(x[i]-_c[i])*(x[i]-_c[i]);
	return y<b;
}
__device__ bool f2(double * x, int n)
{
	const double _c[] = {2, 2};
	const double b = 16;

	double y = 0.0;
	for(unsigned i=0; i<n; i++) y+=(x[i]-_c[i])*(x[i]-_c[i]);
	return y<b;
}
// ...

/////////////////////////////////////////////////////////
// Искомая функция
__device__ double w(double * x, int n)
{
	const double _c[] = {7.0/3, 10.0/3};

	double y = 0.0;
	for(unsigned i=0; i<n; i++) y+=(x[i]-_c[i])*(x[i]-_c[i]);
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
// Дефолтные значения
const unsigned _n = 2;
const int _md = 100;
const unsigned _m[] = {100, 100};
const double _a[] = {0, 0};
const double _b[] = {100, 100};
const double _e=1e-15;

/////////////////////////////////////////////////////////
// Вычисление вектора индексов координат решётки по номеру узла
// index - номер узла решётки
__device__ void vector_of(unsigned * vector, unsigned long index, unsigned * m, int n)
{
	for(unsigned i=0;i<n;i++)
	{
		vector[i]=index%(1ul+m[i]);
		index/=1ul+m[i];
	}
}

/////////////////////////////////////////////////////////
// Преобразование вектора индексов координат решётки
// в вектор координат точки
// vector - вектор индексов координат решётки
// m - число сегментов по каждому из измерений
// a - вектор минимальных координат точек
// b - вектор максимальных координат точек
__device__ void point_of(double * point, unsigned * vector,
						 unsigned * m, double * a, double * b, int n)
{
	for(unsigned i=0;i<n;i++) point[i]=(a[i]*(m[i]-vector[i])+b[i]*vector[i])/m[i];
}

/////////////////////////////////////////////////////////
// Проверка принадлежности точки области, заданной ограничениями
// x - координаты точки
// f - набор проверочных функций
__device__ bool check(double * x, unsigned n)
{
	return f1(x,n)&&f2(x,n);
}

__device__ void copy(double * x, double * y, unsigned n)
{
	for(unsigned i=0; i<n; i++) x[i] = y[i];
}

__global__ void kernel0(
	unsigned * vPtr,
	double * tPtr,
	double * xPtr,
	double * yPtr,
	bool * ePtr,
	unsigned * m,
	double * a,
	double * b,
	unsigned long total,
	unsigned n)
{
	// Получаем идентификатор нити
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
				::copy(x, t, n);
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

__global__ void kernel1(
	double * x0,
	double * tPtr,
	double * xPtr,
	double * yPtr,
	bool * ePtr,
	unsigned mk,
	double ak,
	double bk,
	unsigned k,
	unsigned n)
{
	// Получаем идентификатор нити
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	double * t = &tPtr[id*n];
	double * x = &xPtr[id];
	double * y = &yPtr[id];
	bool * e = &ePtr[id];

	::copy(t, x0, n);
	*e = false;
	for (unsigned i = blockDim.x*blockIdx.x + threadIdx.x;
		i <= mk;
		i += blockDim.x*gridDim.x) {
			t[k] = (ak*(mk-i)+bk*i)/mk;
			if(!check(t, n)) continue;
			if(!*e) {
				*y = w(t, n);
				*x = t[k];
				*e = true;
				continue;
			}
			double y1 =  w(t, n);
			if(y1 < *y) {
				*x = t[k];
				*y = y1;
			}
	}
}

__global__ void kernel2(
	double * tPtr,
	double * xPtr,
	double * yPtr,
	double * pPtr,
	bool * ePtr,
	double * x1,
	double * x2,
	double l,
	double h,
	int md,
	unsigned n)
{
	// Получаем идентификатор нити
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	double * t = &tPtr[id*n];
	double * x = &xPtr[id*n];
	double * y = &yPtr[id];
	double * p = &pPtr[id];
	bool * e = &ePtr[id];

	*e = false;
	for (int index = blockDim.x*blockIdx.x + threadIdx.x-md;
		index <= md;
		index += blockDim.x*gridDim.x) {
			double pt = (l*(md-index)+h*(md+index))/md/2;
			for(unsigned i=0; i<n; i++) t[i] = x2[i] + (x1[i]-x2[i])*pt;
			if(!check(t, n)) continue;
			if(!*e) {
				*y = w(t, n);
				*p = pt;
				::copy(x, t, n);
				*e = true;
				continue;
			}
			double y1 =  w(t, n);
			if(y1 < *y) {
				::copy(x, t, n);
				*y = y1;
				*p = pt;
			}
	}
}

int main(int argc, char* argv[])
{
	// http://stackoverflow.com/questions/2236197/what-is-the-easiest-way-to-initialize-a-stdvector-with-hardcoded-elements

	unsigned n=_n;
	double e=_e;
	int md = _md;
	thrust::host_vector<unsigned> hm(_m, _m + sizeof(_m) / sizeof(_m[0]) );
	thrust::host_vector<double> ha(_a, _a + sizeof(_a) / sizeof(_a[0]) );
	thrust::host_vector<double> hb(_b, _b + sizeof(_b) / sizeof(_b[0]) );

	char * input_file_name = NULL;
	char * output_file_name = NULL;

	int gridSize = 0;
	int blockSize = 0;

	// Поддержка кириллицы в консоли Windows
	// Функция setlocale() имеет два параметра, первый параметр - тип категории локали, в нашем случае LC_TYPE - набор символов, второй параметр — значение локали. 
	// Вместо второго аргумента можно писать "Russian", или оставлять пустые двойные кавычки, тогда набор символов будет такой же как и в ОС.
	setlocale(LC_ALL, "");

	for(int i=1; i<argc; i++)
	{
		if(strcmp(argv[i],"-help")==0) 
		{
			std::cout << "Usage :\t" << argv[0] << " [...] [g <gridSize>] [b <blockSize>] [-input <inputfile>] [-output <outputfile>]" << std::endl;
			std::cout << "Алгоритм циклического покоординатного спуска" << std::endl;
			std::cout << "Используя алгоритм одномерной оптимизации по направлению" << std::endl;
			std::cout << "(Алгоритм деления значений аргумента функции)" << std::endl;
			//			std::cout << "\t-n <размерность пространства>" << std::endl;
			std::cout << "\t-m <число сегментов по каждому из измерений>" << std::endl;
			std::cout << "\t-md <число сегментов по вычисленному направлению>" << std::endl;
			std::cout << "\t-a <минимальные координаты по каждому из измерений>" << std::endl;
			std::cout << "\t-b <максимальные координаты по каждому из измерений>" << std::endl;
			std::cout << "\t-e <точность вычислений>" << std::endl;
			std::cout << "\t-ask/noask" << std::endl;
			std::cout << "\t-trace/notrace" << std::endl;
		}
		else if(strcmp(argv[i],"-ask")==0) ask_mode = ASK;
		else if(strcmp(argv[i],"-noask")==0) ask_mode = NOASK;
		else if(strcmp(argv[i],"-trace")==0) trace_mode = TRACE;
		else if(strcmp(argv[i],"-notrace")==0) trace_mode = NOTRACE;
		//		else if(strcmp(argv[i],"-n")==0) n = atoi(argv[++i]);
		else if(strcmp(argv[i],"-e")==0) e = atof(argv[++i]);
		else if(strcmp(argv[i],"-md")==0) md = atoi(argv[++i]);
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
		//  std::cout << "Введите размерность пространства:"<< std::endl; std::cin >> n;

		std::cout << "Введите число сегментов по каждому из измерений m[" << n << "]:"<< std::endl;
		hm.clear();
		for(unsigned i=0;i<n;i++)
		{
			int x; 
			std::cin >> x;
			hm.push_back(x);
		}

		std::cout << "Введите число сегментов по вычисленному направлению:"<< std::endl;
		std::cin >> md;

		std::cout << "Введите минимальные координаты по каждому из измерений a[" << n << "]:"<< std::endl;
		ha.clear();
		for(unsigned i=0;i<n;i++)
		{
			double x; 
			std::cin >> x;
			ha.push_back(x);
		}

		std::cout << "Введите максимальные координаты по каждому из измерений b[" << n << "]:"<< std::endl;
		hb.clear();
		for(unsigned i=0;i<n;i++)
		{
			double x; 
			std::cin >> x;
			hb.push_back(x);
		}

		std::cout << "Введите точность вычислений:"<< std::endl; std::cin >> e;
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
	std::cout << "Размерность пространства : " << n << std::endl;
	std::cout << "Число сегментов          : "; for(unsigned i=0;i<hm.size();i++) std::cout << hm[i] << " "; std::cout << "+ " << md; std::cout << std::endl; 
	std::cout << "Минимальные координаты   : "; for(unsigned i=0;i<ha.size();i++) std::cout << ha[i] << " "; std::cout << std::endl; 
	std::cout << "Максимальные координаты  : "; for(unsigned i=0;i<hb.size();i++) std::cout << hb[i] << " "; std::cout << std::endl; 
	std::cout << "Точность вычислений      : " << e << std::endl;

	thrust::device_vector<unsigned> m(hm);
	thrust::device_vector<double> a(ha);
	thrust::device_vector<double> b(hb);

	for(unsigned i=0;i<m.size();i++) assert(m[i]>2);

	// Алгоритм
	unsigned mMax = thrust::max((unsigned)md,thrust::reduce(m.begin(), m.end(), 0u, max_functor<unsigned>()));

	// Определим оптимальное разбиения на процессы, нити

	int blocks = (gridSize > 0)? gridSize : thrust::min(15, (int)pow(mMax+1,0.333333));
	int threads = (blockSize > 0)? blockSize : thrust::min(15, (int)pow(mMax+1,0.333333));

	// Аллокируем память для параллельных вычислений
	thrust::device_vector<unsigned> vArray(blocks*threads*n);
	thrust::device_vector<double> tArray(blocks*threads*n);
	thrust::device_vector<double> xArray(blocks*threads*n);
	thrust::device_vector<double> yArray(blocks*threads);
	thrust::device_vector<double> pArray(blocks*threads);
	thrust::device_vector<bool> eArray(blocks*threads);
	thrust::host_vector<double> hyArray(blocks*threads);
	thrust::host_vector<double> hpArray(blocks*threads);

	unsigned * vPtr = thrust::raw_pointer_cast(&vArray[0]);
	double * tPtr = thrust::raw_pointer_cast(&tArray[0]);
	double * xPtr = thrust::raw_pointer_cast(&xArray[0]);
	double * yPtr = thrust::raw_pointer_cast(&yArray[0]);
	double * pPtr = thrust::raw_pointer_cast(&pArray[0]);
	bool * ePtr = thrust::raw_pointer_cast(&eArray[0]);
	unsigned * mPtr = thrust::raw_pointer_cast(&m[0]);
	double * aPtr = thrust::raw_pointer_cast(&a[0]);
	double * bPtr = thrust::raw_pointer_cast(&b[0]);

	thrust::host_vector<double> ht(n);
	thrust::host_vector<double> hx(n);
	thrust::device_vector<double> x1(n);
	thrust::device_vector<double> x2(n);
	double diameter = ::distance(a,b);

	// Алгоритм

	thrust::device_vector<double> x(n);
	double y;
	double * xPtr0 = thrust::raw_pointer_cast(&x[0]);
	double * xPtr1 = thrust::raw_pointer_cast(&x1[0]);
	double * xPtr2 = thrust::raw_pointer_cast(&x2[0]);

	while(true)
	{
		unsigned long total=total_of(m);

		kernel0<<< blocks, threads >>>(vPtr, tPtr, xPtr, yPtr, ePtr, mPtr, aPtr, bPtr, total, n);
		thrust::copy(yArray.begin(), yArray.end(), hyArray.begin());

		// Находим первую точку в области, заданной ограничениями
		auto it = thrust::find(eArray.begin(), eArray.end(), true);
		if(it>=eArray.end())
		{
			for(unsigned i=0; i<n; i++) m[i]<<=1u;
			continue;
		}

		int index = thrust::distance(eArray.begin(), it);
		y=hyArray[index];
		thrust::copy(&xArray[index*n], &xArray[index*n+n], x.begin());

		while((it = thrust::find(it, eArray.end(), true)) < eArray.end())
		{
			int index = thrust::distance(eArray.begin(), it++);
			double y1=hyArray[index];
			if(y<y1) continue;
			y=y1;
			thrust::copy(&xArray[index*n], &xArray[index*n+n], x.begin());
		}
		break;
	}

	while(true)
	{
		// Находим следующую точку в области, заданной ограничениями
		// Используя алгоритм одномерной оптимизации по направлению

		thrust::copy(x.begin(), x.end(), x1.begin()); // Сохранение значения последней точки

		// Цикл по измерениям
		for(unsigned k=0; k<n; k++)
		{
			// Алгоритм одномерной оптимизации по направлению
			double ak = a[k];
			double bk = b[k];
			unsigned mk = m[k];
			while(true)
			{
				kernel1<<< blocks, threads >>>(xPtr0, tPtr, xPtr, yPtr, ePtr, mk, ak, bk, k, n);
				thrust::copy(yArray.begin(), yArray.end(), hyArray.begin());

				// Находим первую точку в области, заданной ограничениями
				auto it = thrust::find(eArray.begin(), eArray.end(), true);

				assert(it<eArray.end());

				int index = thrust::distance(eArray.begin(), it++);
				y=hyArray[index];
				thrust::copy(&xArray[index], &xArray[index+1], &x[k]);

				while((it = thrust::find(it, eArray.end(), true)) < eArray.end())
				{
					int index = thrust::distance(eArray.begin(), it++);
					if(index>mk) break;
					double y1=hyArray[index];
					if(y<y1) continue;
					y=y1;
					thrust::copy(&xArray[index], &xArray[index+1], &x[k]);
				}

				if(thrust::max(ak-bk,bk-ak)<e) break;
				double aa = ak;
				double bb = bk;
				double xx = x[k];
				ak=thrust::max(aa,xx-(bb-aa)/mk);
				bk=thrust::min(bb,xx+(bb-aa)/mk);
			}
		}

		thrust::copy(x.begin(), x.end(), x2.begin()); // Сохранение значения последней точки
		double l = -diameter;
		double h = diameter;

		// Находим следующую точку в области, заданной ограничениями
		// Используя алгоритм одномерной оптимизации по направлению x2->x1
		while(true)
		{
			kernel2<<< blocks, threads >>>(tPtr, xPtr, yPtr, pPtr, ePtr, xPtr1, xPtr2, l, h, md, n);
			thrust::copy(yArray.begin(), yArray.end(), hyArray.begin());
			thrust::copy(pArray.begin(), pArray.end(), hpArray.begin());

			// Находим первую точку в области, заданной ограничениями
			auto it = thrust::find(eArray.begin(), eArray.end(), true);

			assert(it<eArray.end());

			int index = thrust::distance(eArray.begin(), it++);
			y=hyArray[index];
			double p=hpArray[index];
			thrust::copy(&xArray[index*n], &xArray[index*n+n], x.begin());

			while((it = thrust::find(it, eArray.end(), true)) < eArray.end())
			{
				int index = thrust::distance(eArray.begin(), it++);
				double y1=hyArray[index];
				if(y<y1) continue;
				y=y1;
				p=hpArray[index];
				thrust::copy(&xArray[index*n], &xArray[index*n+n], x.begin());
			}
			if(h-l<e) break;
			double ll = l;
			double hh = h;
			l=thrust::max(ll,p-(hh-ll)/md);
			h=thrust::min(hh,p+(hh-ll)/md);
		}

		if(trace_mode==TRACE) {
			thrust::copy(x.begin(), x.end(), hx.begin());
			for(unsigned i=0;i<hx.size();i++) std::cout << hx[i] << " "; 
		}
		if(trace_mode==TRACE) std::cout << "-> " << y << std::endl; 

		if(delta(x,x1)<e) break;
	}

	thrust::copy(x.begin(), x.end(), hx.begin());
	std::cout << "Точка минимума           : "; for(unsigned i=0;i<hx.size();i++) std::cout << hx[i] << " "; std::cout << std::endl; 
	std::cout << "Минимальное значение     : " << y << std::endl; 

	getchar();
	getchar();

	return 0;
}

