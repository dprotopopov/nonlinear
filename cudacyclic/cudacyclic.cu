// Алгоритм циклического покоординатного спуска
// Используя алгоритм одномерной оптимизации по направлению

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

typedef bool (check_func)(thrust::device_vector<double> x); // Профиль проверочной функции
typedef double (value_func)(thrust::device_vector<double> x); // Профиль искомой функции

double delta(thrust::device_vector<double> x,thrust::device_vector<double> y);
unsigned long total_of(thrust::device_vector<unsigned> m);
thrust::device_vector<unsigned> vector_of(unsigned long index, thrust::device_vector<unsigned> m);
thrust::device_vector<double> point_of(thrust::device_vector<unsigned> vector,
	thrust::device_vector<unsigned> m, thrust::device_vector<double> a, thrust::device_vector<double> b);

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
bool f1(thrust::device_vector<double> x)
{
	const double _c[] = {1, 1};
	const double b = 16;

	thrust::device_vector<double> c(_c, _c + sizeof(_c) / sizeof(_c[0]) );
	thrust::device_vector<double> sub(thrust::max(x.size(),c.size()));
	thrust::device_vector<double> square(thrust::max(x.size(),c.size()));
	assert(x.size()==c.size());
	thrust::transform(x.begin(), x.end(), c.begin(), sub.begin(), sub_functor<double>());
	thrust::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>());
	double y=thrust::reduce(square.begin(), square.end(), 0.0, add_functor<double>());
	return y<b;
}
bool f2(thrust::device_vector<double> x)
{
	const double _c[] = {2, 2};
	const double b = 16;

	thrust::device_vector<double> c(_c, _c + sizeof(_c) / sizeof(_c[0]) );
	thrust::device_vector<double> sub(thrust::max(x.size(),c.size()));
	thrust::device_vector<double> square(thrust::max(x.size(),c.size()));
	assert(x.size()==c.size());
	thrust::transform(x.begin(), x.end(), c.begin(), sub.begin(), sub_functor<double>());
	thrust::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>());
	double y=thrust::reduce(square.begin(), square.end(), 0.0, add_functor<double>());
	return y<b;
}
// ...

/////////////////////////////////////////////////////////
// Искомая функция
double w(thrust::device_vector<double> x)
{
	const double _c[] = {2, 3};

	thrust::device_vector<double> c(_c, _c + sizeof(_c) / sizeof(_c[0]) );
	thrust::device_vector<double> sub(thrust::max(x.size(),c.size()));
	thrust::device_vector<double> square(thrust::max(x.size(),c.size()));
	assert(x.size()==c.size());
	thrust::transform(x.begin(), x.end(), c.begin(), sub.begin(), sub_functor<double>());
	thrust::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>());
	return thrust::reduce(square.begin(), square.end(), 0.0, add_functor<double>());
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
static const unsigned _n = 2;
static const unsigned _m[] = {100, 100};
static const double _a[] = {0, 0};
static const double _b[] = {100, 100};
static check_func *_f[]  = {&f1,&f2};
static const double _e=1e-15;


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
// Вычисление вектора индексов координат решётки по номеру узла
// index - номер узла решётки
thrust::device_vector<unsigned> vector_of(unsigned long index, thrust::device_vector<unsigned> m)
{
	thrust::device_vector<unsigned> vector;
	for(unsigned i=0;i<m.size();i++)
	{
		vector.push_back(index%(1ul+m[i]));
		index/=1ul+m[i];
	}
	return vector;
}

/////////////////////////////////////////////////////////
// Преобразование вектора индексов координат решётки
// в вектор координат точки
// vector - вектор индексов координат решётки
// m - число сегментов по каждому из измерений
// a - вектор минимальных координат точек
// b - вектор максимальных координат точек
thrust::device_vector<double> point_of(thrust::device_vector<unsigned> vector,
												  thrust::device_vector<unsigned> m,
												  thrust::device_vector<double> a,
												  thrust::device_vector<double> b)
{
	thrust::device_vector<double> point(m.size());
	for(unsigned i=0;i<m.size();i++) point[i]=(a[i]*(m[i]-vector[i])+b[i]*vector[i])/m[i];
	return point;
}

/////////////////////////////////////////////////////////
// Вычисление числа узлов решётки
// m - число сегментов по каждому из измерений
unsigned long total_of(thrust::device_vector<unsigned> m)
{
	return thrust::transform_reduce(m.begin(), m.end(), inc_functor<unsigned>(), 1UL, mul_functor<unsigned long>());
}

/////////////////////////////////////////////////////////
// Проверка принадлежности точки области, заданной ограничениями
// x - координаты точки
// f - набор проверочных функций
bool check(thrust::device_vector<double> x, thrust::device_vector<check_func *> f)
{
	bool b=true;
	for(unsigned i=0;b&&i<f.size();i++)	b=(*f[i])(x);
	return b;
}


int main(int argc, char* argv[])
{
  // http://stackoverflow.com/questions/2236197/what-is-the-easiest-way-to-initialize-a-stdvector-with-hardcoded-elements

	unsigned n=_n;
	double e=_e;
	thrust::host_vector<unsigned> hm(_m, _m + sizeof(_m) / sizeof(_m[0]) );
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
			std::cout << "Алгоритм циклического покоординатного спуска" << std::endl;
			std::cout << "Используя алгоритм одномерной оптимизации по направлению" << std::endl;
			std::cout << "(Алгоритм деления значений аргумента функции)" << std::endl;
			//			std::cout << "\t-n <размерность пространства>" << std::endl;
			std::cout << "\t-m <число сегментов по каждому из измерений>" << std::endl;
			std::cout << "\t-a <минимальные координаты по каждому из измерений>" << std::endl;
			std::cout << "\t-b <максимальные координаты по каждому из измерений>" << std::endl;
			std::cout << "\t-e <очность вычислений>" << std::endl;
			std::cout << "\t-ask/noask" << std::endl;
			std::cout << "\t-trace/notrace" << std::endl;
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

	int major = THRUST_MAJOR_VERSION;
	int minor = THRUST_MINOR_VERSION;

	std::cout << "Thrust v" << major << "." << minor << std::endl;
	std::cout << "Размерность пространства : " << n << std::endl;
	std::cout << "Число сегментов          : "; for(unsigned i=0;i<hm.size();i++) std::cout << hm[i] << " "; std::cout << std::endl; 
	std::cout << "Минимальные координаты   : "; for(unsigned i=0;i<ha.size();i++) std::cout << ha[i] << " "; std::cout << std::endl; 
	std::cout << "Максимальные координаты  : "; for(unsigned i=0;i<hb.size();i++) std::cout << hb[i] << " "; std::cout << std::endl; 
	std::cout << "Точность вычислений      : " << e << std::endl;

	for(unsigned i=0;i<hm.size();i++) assert(hm[i]>2);

// Алгоритм
	thrust::device_vector<unsigned> m(hm);
	thrust::device_vector<double> a(ha);
	thrust::device_vector<double> b(hb);
	thrust::device_vector<check_func *> f(hf);


	// Находим первую точку в области, заданной ограничениями
	thrust::device_vector<double> x;
	double y;
	while(true)
	{
		unsigned long total=total_of(m);
		unsigned long index=0;
		while(index<total)
		{
			x = point_of(vector_of(index++, m), m, a, b);
			if(check(x,f)) break;
		}
		if(index>=total)
		{
			for(int i=0; i<n; i++) m[i]*=2;
			continue;
		}
		y = (*w)(x);
		break;
	}

	while(true)
	{
		// Находим следующую точку в области, заданной ограничениями
		// Используя алгоритм одномерной оптимизации по направлению
		
		thrust::device_vector<double> x1(x); // Сохранение значения последней точки

		// Цикл по измерениям
		for(int k=0; k<n; k++)
		{
			// Алгоритм одномерной оптимизации по направлению
			double ak = a[k];
			double bk = b[k];
			unsigned mk = m[k];
			while(true)
			{
				thrust::device_vector<double> xk(x);
				for(int i=0; i<=mk; i++)
				{
					xk[k] = (ak*(mk-i)+bk*i)/mk;
					if(!check(xk,f)) continue;
					double yk = (*w)(xk);
					if(yk>y) continue;
					y = yk;
					x[k] = xk[k];
				}
				if(thrust::max(ak-bk,bk-ak)<e) break;
				double aa = ak;
				double bb = bk;
				double xx = x[k];
				ak=thrust::max(aa,xx-(bb-aa)/mk);
				bk=thrust::min(bb,xx+(bb-aa)/mk);
			}
		}

		if(trace_mode==TRACE) {
			thrust::host_vector<double> hx(x);
			for(unsigned i=0;i<hx.size();i++) std::cout << hx[i] << " "; 
		}
		if(trace_mode==TRACE) std::cout << "-> " << y << std::endl; 

		if(delta(x,x1)<e) break;
	}

	thrust::host_vector<double> hx(x);
	std::cout << "Точка минимума           : "; for(unsigned i=0;i<hx.size();i++) std::cout << hx[i] << " "; std::cout << std::endl; 
	std::cout << "Минимальное значение     : " << y << std::endl; 

	getchar();
	getchar();

	return 0;
}

