// Алгоритм многомерной оптимизации с использованием метода решёток

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

typedef bool (check_func)(thrust::device_vector<double> &x); // Профиль проверочной функции
typedef double (value_func)(thrust::device_vector<double> &x); // Профиль искомой функции

double module(thrust::device_vector<double> &x);
double delta(thrust::device_vector<double> &x, thrust::device_vector<double> &y);
unsigned long total_of(thrust::device_vector<size_t> &m);
void vector_of(thrust::device_vector<unsigned> &vector, unsigned long index, thrust::device_vector<size_t> &m);
void point_of(thrust::device_vector<double> &point, thrust::device_vector<unsigned> &vector, thrust::device_vector<size_t> &m, thrust::device_vector<double> &a, thrust::device_vector<double> &b);
bool check(thrust::device_vector<double> &x, thrust::device_vector<check_func *> &f, thrust::device_vector<double> &a, thrust::device_vector<double> &b);

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
// Набор проверочных функций
bool f1(thrust::device_vector<double> &x)
{
	static const double _c[] = {0, 0};
	static const double b = 500;

	thrust::device_vector<double> c(_c, _c + sizeof(_c) / sizeof(_c[0]) );
	thrust::device_vector<double> sub(thrust::max(x.size(),c.size()));
	thrust::device_vector<double> square(thrust::max(x.size(),c.size()));
	assert(x.size()==c.size());
	thrust::transform(x.begin(), x.end(), c.begin(), sub.begin(), sub_functor<double>());
	thrust::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>());
	double y=thrust::reduce(square.begin(), square.end(), 0.0, add_functor<double>());
	return y<b*b;
}
bool f2(thrust::device_vector<double> &x)
{
	static const double _c[] = {100, 100};
	static const double b = 500;

	thrust::device_vector<double> c(_c, _c + sizeof(_c) / sizeof(_c[0]) );
	thrust::device_vector<double> sub(thrust::max(x.size(),c.size()));
	thrust::device_vector<double> square(thrust::max(x.size(),c.size()));
	assert(x.size()==c.size());
	thrust::transform(x.begin(), x.end(), c.begin(), sub.begin(), sub_functor<double>());
	thrust::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>());
	double y=thrust::reduce(square.begin(), square.end(), 0.0, add_functor<double>());
	return y<b*b;
}
// ...

/////////////////////////////////////////////////////////
// Искомая функция
double w(thrust::device_vector<double> &x)
{
	const double _c0[] = {0, 0};
	const double _c1[] = {150, 180};
	const double _c2[] = {240, 200};
	const double _c3[] = {260, 90};
	const double _q[] = {3040, 1800, 800, 1200};

	thrust::device_vector<double> c0(_c0, _c0 + sizeof(_c0) / sizeof(_c0[0]) );
	thrust::device_vector<double> c1(_c1, _c1 + sizeof(_c1) / sizeof(_c1[0]) );
	thrust::device_vector<double> c2(_c2, _c2 + sizeof(_c2) / sizeof(_c2[0]) );
	thrust::device_vector<double> c3(_c3, _c3 + sizeof(_c3) / sizeof(_c3[0]) );
	thrust::device_vector<double> q(_q, _q + sizeof(_q) / sizeof(_q[0]) );
	thrust::device_vector<double> d;
	thrust::device_vector<double> sub(x.size());
	thrust::device_vector<double> square(x.size());
	thrust::device_vector<double> mul(q.size());
	assert(x.size()==c0.size());
	assert(x.size()==c1.size());
	assert(x.size()==c2.size());
	assert(x.size()==c3.size());
	thrust::transform(x.begin(), x.end(), c0.begin(), sub.begin(), sub_functor<double>());
	thrust::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>());
	d.push_back(std::sqrt(thrust::reduce(square.begin(), square.end(), 0.0, add_functor<double>())));
	thrust::transform(x.begin(), x.end(), c1.begin(), sub.begin(), sub_functor<double>());
	thrust::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>());
	d.push_back(std::sqrt(thrust::reduce(square.begin(), square.end(), 0.0, add_functor<double>())));
	thrust::transform(x.begin(), x.end(), c2.begin(), sub.begin(), sub_functor<double>());
	thrust::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>());
	d.push_back(std::sqrt(thrust::reduce(square.begin(), square.end(), 0.0, add_functor<double>())));
	thrust::transform(x.begin(), x.end(), c3.begin(), sub.begin(), sub_functor<double>());
	thrust::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>());
	d.push_back(std::sqrt(thrust::reduce(square.begin(), square.end(), 0.0, add_functor<double>())));
	thrust::transform(q.begin(), q.end(), d.begin(), mul.begin(), mul_functor<double>());
	return thrust::reduce(mul.begin(), mul.end(), 0.0, add_functor<double>());
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
// Дефолтные значения
static const unsigned _count = 1;
static const size_t _n = 2;
static const size_t _m[] = {3, 3};
static const double _a[] = {0, 0};
static const double _b[] = {1000, 1000};
static check_func *_f[]  = {&f1,&f2};
static const double _e=1e-8;

/////////////////////////////////////////////////////////
// Вычисление модуля вектора
double module(thrust::device_vector<double> &x)
{
	return thrust::transform_reduce(x.begin(), x.end(), abs_functor<double>(), 0.0, max_functor<double>());
}

/////////////////////////////////////////////////////////
// Вычисление растояния между двумя векторами координат
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
// Вычисление вектора индексов координат решётки по номеру узла
// vector - вектор индексов координат решётки
// index - номер узла решётки
// m - число сегментов по каждому из измерений
void vector_of(thrust::device_vector<unsigned> &vector, unsigned long index, thrust::device_vector<size_t> &m)
{
	for(size_t i=0;i<m.size();i++)
	{
		unsigned long m1 = 1ul+m[i];
		vector[i]=index%m1;
		index/=m1;
	}
}

/////////////////////////////////////////////////////////
// Преобразование вектора индексов координат решётки
// в вектор координат точки
// vector - вектор индексов координат решётки
// m - число сегментов по каждому из измерений
// a - вектор минимальных координат точек
// b - вектор максимальных координат точек
void point_of(thrust::device_vector<double> &point, thrust::device_vector<unsigned> &vector, thrust::device_vector<size_t> &m, thrust::device_vector<double> &a, thrust::device_vector<double> &b)
{
	for(size_t i=0;i<m.size();i++) point[i]=(a[i]*(m[i]-vector[i])+b[i]*vector[i])/m[i];
}

/////////////////////////////////////////////////////////
// Вычисление числа узлов решётки
// m - число сегментов по каждому из измерений
unsigned long total_of(thrust::device_vector<size_t> &m)
{
	return thrust::transform_reduce(m.begin(), m.end(), inc_functor<size_t>(), 1UL, mul_functor<unsigned long>());
}

/////////////////////////////////////////////////////////
// Проверка принадлежности точки области, заданной ограничениями
// x - координаты точки
// f - набор проверочных функций
// a - вектор минимальных координат точек
// b - вектор максимальных координат точек
bool check(thrust::device_vector<double> &x, thrust::device_vector<check_func *> &f, thrust::device_vector<double> &a, thrust::device_vector<double> &b)
{
	for(size_t i=0;i<a.size();i++) if(x[i]<a[i]&&x[i]<b[i]) return false;
	for(size_t i=0;i<b.size();i++) if(x[i]>a[i]&&x[i]>b[i]) return false;
	for(size_t i=0;i<f.size();i++) if(!(*f[i])(x)) return false;
	return true;
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
	thrust::host_vector<check_func *> hf(_f, _f + sizeof(_f) / sizeof(_f[0]) );

	char * input_file_name = NULL;
	char * output_file_name = NULL;

	// Поддержка кириллицы в консоли Windows
	// Функция setlocale() имеет два параметра, первый параметр - тип категории локали, в нашем случае LC_TYPE - набор символов, второй параметр — значение локали. 
	// Вместо второго аргумента можно писать "Russian", или оставлять пустые двойные кавычки, тогда набор символов будет такой же как и в ОС.
	setlocale(LC_ALL, "");

	for(int i=1; i<argc; i++)
	{
		if(strcmp(argv[i],"-help")==0) 
		{
			std::cout << "Usage :\t" << argv[0] << " [...] [-input <inputfile>] [-output <outputfile>]" << std::endl;
			std::cout << "Алгоритм многомерной оптимизации с использованием метода решёток" << std::endl;
			std::cout << "Алгоритм деления значений аргумента функции" << std::endl;
			//			std::cout << "\t-n <размерность пространства>" << std::endl;
			std::cout << "\t-c <количество повторений алгоритма для замера времени>" << std::endl;
			std::cout << "\t-m <число сегментов по каждому из измерений>" << std::endl;
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
	}

	if(input_file_name!=NULL) freopen(input_file_name,"r",stdin);
	if(output_file_name!=NULL) freopen(output_file_name,"w",stdout);


	if(ask_mode == ASK)
	{
		//  std::cout << "Введите размерность пространства:"<< std::endl; std::cin >> n;

		std::cout << "Введите число сегментов по каждому из измерений m[" << n << "]:"<< std::endl;
		hm.clear();
		for(size_t i=0;i<n;i++)
		{
			int x; 
			std::cin >> x;
			hm.push_back(x);
		}

		std::cout << "Введите минимальные координаты по каждому из измерений a[" << n << "]:"<< std::endl;
		ha.clear();
		for(size_t i=0;i<n;i++)
		{
			double x; 
			std::cin >> x;
			ha.push_back(x);
		}

		std::cout << "Введите максимальные координаты по каждому из измерений b[" << n << "]:"<< std::endl;
		hb.clear();
		for(size_t i=0;i<n;i++)
		{
			double x; 
			std::cin >> x;
			hb.push_back(x);
		}

		std::cout << "Введите точность вычислений:"<< std::endl; std::cin >> e;
		std::cout << "Введите количество повторений алгоритма для замера времени:"<< std::endl; std::cin >> count;
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

// Алгоритм
	clock_t time = clock();

	thrust::device_vector<size_t> m(hm);
	thrust::device_vector<check_func *> f(hf);
	thrust::device_vector<double> a(ha);
	thrust::device_vector<double> b(hb);

	thrust::host_vector<double> hx(n);
	thrust::device_vector<unsigned> v(n);
	thrust::device_vector<double> x(n);
	thrust::device_vector<double> x1(n);
	thrust::device_vector<double> a1(n);
	thrust::device_vector<double> b1(n);
	double y;

	if(trace_mode==TRACE&&count==1) std::cout << "for #1" << std::endl; 
	for(unsigned s=0; s<count; s++)
	{
		thrust::copy(a.begin(), a.end(), a1.begin());
		thrust::copy(b.begin(), b.end(), b1.begin());

		if(trace_mode==TRACE&&count==1) std::cout << "while #1" << std::endl; 
		while(true)
		{
			// Находим первую точку в области, заданной ограничениями
			unsigned long total=total_of(m);

			unsigned long index=0;
			while(index<total)
			{
				vector_of(v, index++, m);
				point_of(x, v, m, a1, b1);
				if(check(x, f, a1, b1)) break;
			}
			if(index>=total)
			{
				for(size_t i=0; i<n; i++) m[i]<<=1u;
				continue;
			}
			y=(*w)(x);

			if(trace_mode==TRACE&&count==1) {
				thrust::copy(x.begin(), x.end(), hx.begin());
				for(size_t i=0;i<hx.size();i++) std::cout << hx[i] << " "; 
			}
			if(trace_mode==TRACE&&count==1) std::cout << "-> " << y << std::endl; 

			while(index<total)
			{
				vector_of(v, index++, m);
				point_of(x1, v, m, a1, b1);
				if(!check(x1, f, a1, b1)) continue;
				double y1 = (*w)(x1);
				if(y1>y) continue;
				thrust::copy(x1.begin(), x1.end(), x.begin());
				y=y1;

				if(trace_mode==TRACE&&count==1) {
					thrust::host_vector<double> hx(x);
					for(size_t i=0;i<hx.size();i++) std::cout << hx[i] << " "; 
				}
				if(trace_mode==TRACE&&count==1) std::cout << "-> " << y << std::endl; 
			}

			double dd = delta(a1, b1);
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
	std::cout << "Исполняемый файл         : " << argv[0] << std::endl;
	std::cout << "Размерность пространства : " << n << std::endl;
	std::cout << "Число сегментов          : "; for(size_t i=0;i<hm.size();i++) std::cout << hm[i] << " "; std::cout << std::endl; 
	std::cout << "Минимальные координаты   : "; for(size_t i=0;i<ha.size();i++) std::cout << ha[i] << " "; std::cout << std::endl; 
	std::cout << "Максимальные координаты  : "; for(size_t i=0;i<hb.size();i++) std::cout << hb[i] << " "; std::cout << std::endl; 
	std::cout << "Точность вычислений      : " << e << std::endl;
	std::cout << "Точка минимума           : "; for(size_t i=0;i<hx.size();i++) std::cout << hx[i] << " "; std::cout << std::endl; 
	std::cout << "Минимальное значение     : " << y << std::endl; 
	std::cout << "Время вычислений (сек.)  : " << seconds << std::endl; 

	getchar();
	getchar();

	return 0;
}

