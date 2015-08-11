// Алгоритм Хука и Дживса с использованием одномерной минимизации
// Базара М., Шетти К.
// Нелинейное программирование. Теория и алгоритмы:
// Пер. с англ. - М.: Мир, 1982.
// 583 с.

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <clocale>
#include <functional> 
#include <numeric>      // std::accumulate
#include <cctype>
#include <locale>
#include <assert.h>

using namespace std;

typedef bool (check_func)(std::vector<double> &x); // Профиль проверочной функции
typedef double (value_func)(std::vector<double> &x); // Профиль искомой функции

bool check(std::vector<double> &x, std::vector<check_func *> &f, std::vector<double> &a, std::vector<double> &b);
double module(std::vector<double> &x);
double delta(std::vector<double> &x, std::vector<double> &y);
double distance(std::vector<double> &x, std::vector<double> &y);
unsigned long total_of(std::vector<size_t> &m);
void vector_of(std::vector<unsigned> &vector, unsigned long index, std::vector<size_t> &m);
void point_of(std::vector<double> &point, std::vector<unsigned> &vector, std::vector<size_t> &m, std::vector<double> &a, std::vector<double> &b);

template <typename T>
T inc_functor(T value)  
{
	return ++value; 
} 

template <typename T>
T square_functor(T value)  
{
	return value*value; 
} 

template <typename T>
T add_functor(T value1, T value2)  
{
	return value1+value2; 
} 

template <typename T>
T sub_functor(T value1, T value2)  
{
	return value1-value2; 
} 

template <typename T>
T mul_functor(T value1, T value2)  
{
	return value1*value2; 
} 

template <typename T>
T abs_functor(T value)  
{
	return std::abs(value); 
} 
template <typename T>
T diff_functor(T value1, T value2)  
{
	return std::abs(value1-value2); 
} 

template <typename T>
T max_functor(T value1, T value2)  
{
	return std::max(value1,value2); 
} 

bool and_functor(const bool value1, const bool value2)  
{
	return value1&&value2; 
} 

/////////////////////////////////////////////////////////
// Набор проверочных функций
bool f1(std::vector<double> &x)
{
	static const double _c[] = {0, 0};
	static const double b = 500;

	std::vector<double> c(_c, _c + sizeof(_c) / sizeof(_c[0]) );
	std::vector<double> sub(std::max(x.size(),c.size()));
	std::vector<double> square(std::max(x.size(),c.size()));
	assert(x.size()==c.size());
	std::transform(x.begin(), x.end(), c.begin(), sub.begin(), sub_functor<double>);
	std::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>);
	double y=std::accumulate(square.begin(), square.end(), 0.0, add_functor<double>);
	return y<b*b;
}
bool f2(std::vector<double> &x)
{
	static const double _c[] = {100, 100};
	static const double b = 500;

	std::vector<double> c(_c, _c + sizeof(_c) / sizeof(_c[0]) );
	std::vector<double> sub(std::max(x.size(),c.size()));
	std::vector<double> square(std::max(x.size(),c.size()));
	assert(x.size()==c.size());
	std::transform(x.begin(), x.end(), c.begin(), sub.begin(), sub_functor<double>);
	std::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>);
	double y=std::accumulate(square.begin(), square.end(), 0.0, add_functor<double>);
	return y<b*b;
}
// ...

/////////////////////////////////////////////////////////
// Искомая функция
double w(std::vector<double> &x)
{
	const double _c0[] = {0, 0};
	const double _c1[] = {150, 180};
	const double _c2[] = {240, 200};
	const double _c3[] = {260, 90};
	const double _q[] = {3040, 1800, 800, 1200};

	std::vector<double> c0(_c0, _c0 + sizeof(_c0) / sizeof(_c0[0]) );
	std::vector<double> c1(_c1, _c1 + sizeof(_c1) / sizeof(_c1[0]) );
	std::vector<double> c2(_c2, _c2 + sizeof(_c2) / sizeof(_c2[0]) );
	std::vector<double> c3(_c3, _c3 + sizeof(_c3) / sizeof(_c3[0]) );
	std::vector<double> q(_q, _q + sizeof(_q) / sizeof(_q[0]) );
	std::vector<double> d;
	std::vector<double> sub(x.size());
	std::vector<double> square(x.size());
	std::vector<double> mul(q.size());
	assert(x.size()==c0.size());
	assert(x.size()==c1.size());
	assert(x.size()==c2.size());
	assert(x.size()==c3.size());
	std::transform(x.begin(), x.end(), c0.begin(), sub.begin(), sub_functor<double>);
	std::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>);
	d.push_back(std::sqrt(std::accumulate(square.begin(), square.end(), 0.0, add_functor<double>)));
	std::transform(x.begin(), x.end(), c1.begin(), sub.begin(), sub_functor<double>);
	std::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>);
	d.push_back(std::sqrt(std::accumulate(square.begin(), square.end(), 0.0, add_functor<double>)));
	std::transform(x.begin(), x.end(), c2.begin(), sub.begin(), sub_functor<double>);
	std::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>);
	d.push_back(std::sqrt(std::accumulate(square.begin(), square.end(), 0.0, add_functor<double>)));
	std::transform(x.begin(), x.end(), c3.begin(), sub.begin(), sub_functor<double>);
	std::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>);
	d.push_back(std::sqrt(std::accumulate(square.begin(), square.end(), 0.0, add_functor<double>)));
	std::transform(q.begin(), q.end(), d.begin(), mul.begin(), mul_functor<double>);
	return std::accumulate(mul.begin(), mul.end(), 0.0, add_functor<double>);
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
static const size_t _md = 20;
static const size_t _m[] = {20, 20};
static const double _a[] = {0, 0};
static const double _b[] = {1000, 1000};
static check_func *_f[]  = {&f1,&f2};
static const double _e=1e-8;


/////////////////////////////////////////////////////////
// Вычисление модуля вектора
double module(std::vector<double> &x)
{
	std::vector<double> y(x.size());
	std::transform(x.begin(), x.end(), y.begin(), abs_functor<double>);
	return std::accumulate(y.begin(), y.end(), 0.0, max_functor<double>);
}

/////////////////////////////////////////////////////////
// Вычисление растояния между двумя векторами координат
double delta(std::vector<double> &x, std::vector<double> &y)
{
	size_t i=std::min(x.size(),y.size());
	std::vector<double> diff(std::max(x.size(),y.size()));
	std::transform(x.begin(), x.begin()+i, y.begin(), diff.begin(), diff_functor<double>);
	std::transform(x.begin()+i, x.end(), diff.begin()+i, abs_functor<double>);
	std::transform(y.begin()+i, y.end(), diff.begin()+i, abs_functor<double>);
	return std::accumulate(diff.begin(), diff.end(), 0.0, max_functor<double>);
}

/////////////////////////////////////////////////////////
// Вычисление растояния между двумя векторами координат
double distance(std::vector<double> &x, std::vector<double> &y)
{
	size_t i=std::min(x.size(),y.size());
	std::vector<double> sub(std::max(x.size(),y.size()));
	std::vector<double> square(std::max(x.size(),y.size()));
	std::transform(x.begin(), x.end(), y.begin(), sub.begin(), sub_functor<double>);
	std::copy(x.begin()+i, x.end(), sub.begin()+i);
	std::copy(y.begin()+i, y.end(), sub.begin()+i);
	std::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>);
	return std::sqrt(std::accumulate(square.begin(), square.end(), 0.0, add_functor<double>));
}

/////////////////////////////////////////////////////////
// Вычисление вектора индексов координат решётки по номеру узла
// vector - вектор индексов координат решётки
// index - номер узла решётки
// m - число сегментов по каждому из измерений
void vector_of(std::vector<unsigned> &vector, unsigned long index, std::vector<size_t> &m)
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
void point_of(std::vector<double> &point, std::vector<unsigned> &vector, std::vector<size_t> &m, std::vector<double> &a, std::vector<double> &b)
{
	for(size_t i=0;i<m.size();i++) point[i]=(a[i]*(m[i]-vector[i])+b[i]*vector[i])/m[i];
}

/////////////////////////////////////////////////////////
// Вычисление числа узлов решётки
// m - число сегментов по каждому из измерений
unsigned long total_of(std::vector<size_t> &m)
{
	std::vector<size_t> m1(m.size());
	std::transform(m.begin(), m.end(), m1.begin(), inc_functor<size_t>);
	return std::accumulate(m1.begin(), m1.end(), 1UL, mul_functor<unsigned long>);
}

/////////////////////////////////////////////////////////
// Проверка принадлежности точки области, заданной ограничениями
// x - координаты точки
// f - набор проверочных функций
// a - вектор минимальных координат точек
// b - вектор максимальных координат точек
bool check(std::vector<double> &x, std::vector<check_func *> &f, std::vector<double> &a, std::vector<double> &b)
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
	size_t md = _md;
	std::vector<size_t> m(_m, _m + sizeof(_m) / sizeof(_m[0]) );
	std::vector<double> a(_a, _a + sizeof(_a) / sizeof(_a[0]) );
	std::vector<double> b(_b, _b + sizeof(_b) / sizeof(_b[0]) );
	std::vector<check_func *> f(_f, _f + sizeof(_f) / sizeof(_f[0]) );

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
			std::cout << "Алгоритм Хука и Дживса с использованием одномерной минимизации" << std::endl;
			std::cout << "(Алгоритм деления значений аргумента функции)" << std::endl;
			//			std::cout << "\t-n <размерность пространства>" << std::endl;
			std::cout << "\t-c <количество повторений алгоритма для замера времени>" << std::endl;
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
		else if(strcmp(argv[i],"-c")==0) count = atoi(argv[++i]);
		else if(strcmp(argv[i],"-md")==0) md = atoi(argv[++i]);
		else if(strcmp(argv[i],"-m")==0) {
			std::istringstream ss(argv[++i]);
			m.clear();
			for(size_t i=0;i<n;i++) m.push_back(atoi(argv[++i]));
		}
		else if(strcmp(argv[i],"-a")==0) {
			a.clear();
			for(size_t i=0;i<n;i++) a.push_back(atof(argv[++i]));
		}
		else if(strcmp(argv[i],"-b")==0) {
			b.clear();
			for(size_t i=0;i<n;i++) b.push_back(atof(argv[++i]));
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
		m.clear();
		for(size_t i=0;i<n;i++)
		{
			int x; std::cin >> x;
			m.push_back(x);
		}

		std::cout << "Введите число сегментов по вычисленному направлению:"<< std::endl; std::cin >> md;

		std::cout << "Введите минимальные координаты по каждому из измерений a[" << n << "]:"<< std::endl;
		a.clear();
		for(size_t i=0;i<n;i++)
		{
			double x; std::cin >> x;
			a.push_back(x);
		}

		std::cout << "Введите максимальные координаты по каждому из измерений b[" << n << "]:"<< std::endl;
		b.clear();
		for(size_t i=0;i<n;i++)
		{
			double x; std::cin >> x;
			b.push_back(x);
		}

		std::cout << "Введите точность вычислений:"<< std::endl; std::cin >> e;
		std::cout << "Введите количество повторений алгоритма для замера времени:"<< std::endl; std::cin >> count;
	}

	for(size_t i=0;i<m.size();i++) assert(m[i]>2);
	assert(md>2);

// Алгоритм
	clock_t time = clock();

	double diameter = ::distance(a,b);

	std::vector<unsigned> v(n);
	std::vector<double> t(n);
	std::vector<double> x(n);
	std::vector<double> x1(n);
	std::vector<double> x2(n);
	double y;

	if(trace_mode==TRACE&&count==1) std::cout << "for #1" << std::endl; 
	for(unsigned s=0; s<count; s++)
	{
		if(trace_mode==TRACE&&count==1) std::cout << "while #1" << std::endl; 
		while(true)
		{
			// Находим первую точку в области, заданной ограничениями
			unsigned long total=total_of(m);
			unsigned long index=0;
			while(index<total)
			{
				vector_of(v, index++, m);
				point_of(x, v, m, a, b);
				if(check(x, f, a, b)) break;
			}
			if(index>=total)
			{
				for(size_t i=0; i<n; i++) m[i]<<=1u;
				continue;
			}
			y = (*w)(x);
			if(trace_mode==TRACE&&count==1) for(size_t i=0;i<x.size();i++) std::cout << x[i] << " "; 
			if(trace_mode==TRACE&&count==1) std::cout << "-> " << y << std::endl; 
			break;
		}

		if(trace_mode==TRACE&&count==1) std::cout << "while #2" << std::endl; 
		while(true)
		{
			// Находим следующую точку в области, заданной ограничениями
			// Используя алгоритм одномерной оптимизации по направлению
			// Направлениями являются оси пространства

			std::copy(x.begin(), x.end(), x1.begin()); // Сохранение значения последней точки

			// Цикл по измерениям
			if(trace_mode==TRACE&&count==1) std::cout << "for #2" << std::endl; 
			for(size_t k=0; k<n; k++)
			{
				// Алгоритм одномерной оптимизации по направлению
				double ak = std::min(a[k],b[k]);
				double bk = std::max(a[k],b[k]);
				size_t mk = m[k];
				std::copy(x.begin(), x.end(), t.begin());
				while(true)
				{
					for(size_t index=0; index<=mk; index++)
					{
						t[k] = (ak*(mk-index)+bk*index)/mk;
						if(!check(t, f, a, b)) continue;
						double yk = (*w)(t);
						if(yk>y) continue;
						y = yk;
						x[k] = t[k];
						if(trace_mode==TRACE&&count==1) for(size_t i=0;i<x.size();i++) std::cout << x[i] << " "; 
						if(trace_mode==TRACE&&count==1) std::cout << "-> " << y << std::endl; 
					}
					double dd = std::max(ak-bk,bk-ak);
					double cc = std::max(std::max(ak,-ak),std::max(-bk,bk));
					if(dd<=cc*e) break;
					double xk = x[k];
					ak=std::max(ak,xk-dd/mk);
					bk=std::min(bk,xk+dd/mk);
				}
			}

			std::copy(x.begin(), x.end(), x2.begin()); // Сохранение значения последней точки

			// Находим следующую точку в области, заданной ограничениями
			// Используя алгоритм одномерной оптимизации по направлению x2->x1
			double l = -diameter;
			double h = diameter;

			if(trace_mode==TRACE&&count==1) std::cout << "while #3" << std::endl; 
			while(true)
			{
				double p = 0;

				for(size_t index=0;index<=md+md;index++)
				{
					double pt = (l*(md+md-index)+h*index)/(md+md);
					for(size_t i=0; i<n; i++) t[i] = x2[i]*(1.0-pt) + x1[i]*pt;
					if(!check(t, f, a, b)) continue;
					double yt = (*w)(t);
					if(yt>y) continue;
					p = pt;
					y = yt;
					std::copy(t.begin(), t.end(), x.begin());
					if(trace_mode==TRACE&&count==1) for(size_t i=0;i<x.size();i++) std::cout << x[i] << " "; 
					if(trace_mode==TRACE&&count==1) std::cout << "-> " << y << std::endl; 
				}
				double dd = std::max(h-l,l-h);
				double cc = std::max(std::max(h,-h),std::max(-l,l));
				if(dd<=cc*e) break;
				double ll = l;
				double hh = h;
				l=std::max(ll,p-dd/md);
				h=std::min(hh,p+dd/md);
			}

			double dd = delta(x,x1);
			double cc = std::max(module(x),module(x1));
			if(dd<=cc*e) break;
		}
	}

	time = clock() - time;
	double seconds = ((double)time)/CLOCKS_PER_SEC/count;

	std::cout << "Исполняемый файл         : " << argv[0] << std::endl;
	std::cout << "Размерность пространства : " << n << std::endl;
	std::cout << "Число сегментов          : "; for(size_t i=0;i<m.size();i++) std::cout << m[i] << " "; std::cout << "+ " << md; std::cout << std::endl; 
	std::cout << "Минимальные координаты   : "; for(size_t i=0;i<a.size();i++) std::cout << a[i] << " "; std::cout << std::endl; 
	std::cout << "Максимальные координаты  : "; for(size_t i=0;i<b.size();i++) std::cout << b[i] << " "; std::cout << std::endl; 
	std::cout << "Точность вычислений      : " << e << std::endl;
	std::cout << "Точка минимума           : "; for(size_t i=0;i<x.size();i++) std::cout << x[i] << " "; std::cout << std::endl; 
	std::cout << "Минимальное значение     : " << y << std::endl; 
	std::cout << "Время вычислений (сек.)  : " << seconds << std::endl; 

	getchar();
	getchar();

	return 0;
}

