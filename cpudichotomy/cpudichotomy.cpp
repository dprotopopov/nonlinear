// Алгоритм многомерной оптимизации с использованием метода решёток

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <numeric>
#include <locale>
#include <assert.h>
#include <fstream>
#include <stack>
#include <functional>
#include <map>

using namespace std;

// http://stackoverflow.com/questions/1494399/how-do-i-search-find-and-replace-in-a-standard-string
void replace_all(std::string& str, const std::string& oldStr, const std::string& newStr)
{
	size_t pos = 0;
	while ((pos = str.find(oldStr, pos)) != std::string::npos)
	{
		str.replace(pos, oldStr.length(), newStr);
		pos += newStr.length();
	}
}


// https://habrahabr.ru/post/216449/
double calculate(const char* argv, double* x, int n)
{
	string expression(argv);
	for (int i = n; i-- > 0;)
	{
		// заменяем идентификаторы x0,...,xN-1 значениями
		std::ostringstream strs1;
		strs1 << "x" << i;
		string xi = strs1.str();
		std::ostringstream strs2;
		strs2 << x[i];
		string vi = strs2.str();
		replace_all(expression, xi, vi);
	}

	stack<double> s;
	stack<pair<int, char>> ops;

	auto p = [&s, &ops](function<double(double, double)>& f)
		{
			double r = s.top();
			s.pop();
			r = f(s.top(), r);
			s.pop();
			s.push(r);
			ops.pop();
		};

	map<char, pair<int, function<double(double, double)>>> m =
		{{'+',{1, [](double a, double b)
				{
					return a + b;
				}}},{'-',{1, [](double a, double b)
				{
					return a - b;
				}}},
			{'*',{2, [](double a, double b)
				{
					return a * b;
				}}},{'/',{2, [](double a, double b)
				{
					return a / b;
				}}}};

	const int order = 2;
	int level = 0;
	for (char* sp = (char*)expression.c_str();; ++sp)
	{
		while (*sp == '(')
		{
			level += order;
			++sp;
		}

		s.push(strtod(sp, &sp));

		while (*sp == ')')
		{
			level -= order;
			++sp;
		}

		if (!*sp)
		{
			while (!ops.empty()) p(m[ops.top().second].second);
			break;
		}

		const int op = m[*sp].first + level;
		while (!ops.empty() && ops.top().first >= op) p(m[ops.top().second].second);

		ops.push(make_pair(op, *sp));
	}

	return s.top();
}

double module(std::vector<double>& x);
double delta(std::vector<double>& x, std::vector<double>& y);
unsigned long total_of(std::vector<size_t>& m);
void vector_of(std::vector<unsigned>& vector, unsigned long index, std::vector<size_t>& m);
void point_of(std::vector<double>& point, std::vector<unsigned>& vector, std::vector<size_t>& m, std::vector<double>& a, std::vector<double>& b);

template <typename T>
T inc_functor(T value)
{
	return ++value;
}

template <typename T>
T square_functor(T value)
{
	return value * value;
}

template <typename T>
T add_functor(T value1, T value2)
{
	return value1 + value2;
}

template <typename T>
T sub_functor(T value1, T value2)
{
	return value1 - value2;
}

template <typename T>
T mul_functor(T value1, T value2)
{
	return value1 * value2;
}

template <typename T>
T abs_functor(T value)
{
	return std::abs(value);
}

template <typename T>
T diff_functor(T value1, T value2)
{
	return std::abs(value1 - value2);
}

template <typename T>
T max_functor(T value1, T value2)
{
	return std::max(value1, value2);
}

bool and_functor(const bool value1, const bool value2)
{
	return value1 && value2;
}

bool or_functor(const bool value1, const bool value2)
{
	return value1 || value2;
}

enum t_ask_mode
{
	NOASK = 0,
	ASK = 1
};

enum t_trace_mode
{
	NOTRACE = 0,
	TRACE = 1
};

t_ask_mode ask_mode = NOASK;
t_trace_mode trace_mode = NOTRACE;

/////////////////////////////////////////////////////////
// Дефолтные значения
static const unsigned _count = 1;
static const size_t _n = 2;
static const size_t _m[] = {20, 20};
static const double _a[] = {0, 0};
static const double _b[] = {1000, 1000};
static const double _f1[] = {0, 0, 500};
static const double _f2[] = {100, 100, 500};
static const double* _f[] = {_f1, _f2};
//static const double _w1[] = {0, 0, 3040};
//static const double _w2[] = {150, 180, 1800};
//static const double _w3[] = {240, 200, 800};
//static const double _w4[] = {260, 90, 1200};
//static const double* _w[] = {_w1, _w2, _w3, _w4};
static const double _e = 1e-8;


/////////////////////////////////////////////////////////
// Вычисление модуля вектора
double module(std::vector<double>& x)
{
	std::vector<double> y(x.size());
	std::transform(x.begin(), x.end(), y.begin(), abs_functor<double>);
	return std::accumulate(y.begin(), y.end(), 0.0, max_functor<double>);
}

/////////////////////////////////////////////////////////
// Вычисление растояния между двумя векторами координат
double delta(std::vector<double>& x, std::vector<double>& y)
{
	size_t i = std::min(x.size(), y.size());
	std::vector<double> diff(std::max(x.size(), y.size()));
	std::transform(x.begin(), x.begin() + i, y.begin(), diff.begin(), diff_functor<double>);
	std::transform(x.begin() + i, x.end(), diff.begin() + i, abs_functor<double>);
	std::transform(y.begin() + i, y.end(), diff.begin() + i, abs_functor<double>);
	return std::accumulate(diff.begin(), diff.end(), 0.0, max_functor<double>);
}

/////////////////////////////////////////////////////////
// Вычисление вектора индексов координат решётки по номеру узла
// vector - вектор индексов координат решётки
// index - номер узла решётки
// m - число сегментов по каждому из измерений
void vector_of(std::vector<unsigned>& vector, unsigned long index, std::vector<size_t>& m)
{
	for (size_t i = 0; i < m.size(); i++)
	{
		unsigned long m1 = 1ul + m[i];
		vector[i] = index % m1;
		index /= m1;
	}
}

/////////////////////////////////////////////////////////
// Преобразование вектора индексов координат решётки
// в вектор координат точки
// vector - вектор индексов координат решётки
// m - число сегментов по каждому из измерений
// a - вектор минимальных координат точек
// b - вектор максимальных координат точек
void point_of(std::vector<double>& point, std::vector<unsigned>& vector, std::vector<size_t>& m, std::vector<double>& a, std::vector<double>& b)
{
	for (size_t i = 0; i < m.size(); i++) point[i] = (a[i] * (m[i] - vector[i]) + b[i] * vector[i]) / m[i];
}

/////////////////////////////////////////////////////////
// Вычисление числа узлов решётки
// m - число сегментов по каждому из измерений
unsigned long total_of(std::vector<size_t>& m)
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
bool check(std::vector<double>& x, std::vector<double>& f, std::vector<double>& a, std::vector<double>& b)
{
	size_t n = x.size();
	std::vector<bool> s(f.size() / (n + 1));
	for (size_t i = 0; i < n; i++) if (x[i] < a[i] && x[i] < b[i]) return false;
	for (size_t i = 0; i < n; i++) if (x[i] > a[i] && x[i] > b[i]) return false;
#pragma omp parallel for
	for (size_t i = 0; i < s.size(); i++)
	{
		std::vector<double> sub(x.size());
		std::vector<double> square(x.size());
		std::transform(x.begin(), x.end(), f.begin() + i * (n + 1), sub.begin(), diff_functor<double>);
		std::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>);
		s[i] = (std::sqrt(std::accumulate(square.begin(), square.end(), 0.0, add_functor<double>)) < f[i * (n + 1) + n]);
	}
	return std::accumulate(s.begin(), s.end(), true, and_functor);
}

/////////////////////////////////////////////////////////
// Искомая функция
//double target(std::vector<double>& x, std::vector<double>& w)
//{
//	size_t n = x.size();
//	std::vector<double> s(w.size() / (n + 1));
//#pragma omp parallel for
//	for (size_t i = 0; i < s.size(); i++)
//	{
//		std::vector<double> sub(x.size());
//		std::vector<double> square(x.size());
//		std::transform(x.begin(), x.end(), w.begin() + i * (n + 1), sub.begin(), diff_functor<double>);
//		std::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>);
//		s[i] = std::sqrt(std::accumulate(square.begin(), square.end(), 0.0, add_functor<double>)) * w[i * (n + 1) + n];
//	}
//	return std::accumulate(s.begin(), s.end(), 0.0, add_functor<double>);
//}


int main(int argc, char* argv[])
{
	// http://stackoverflow.com/questions/2236197/what-is-the-easiest-way-to-initialize-a-stdvector-with-hardcoded-elements

	unsigned count = _count;
	size_t n = _n;
	double e = _e;
	std::vector<size_t> m(_m, _m + sizeof(_m) / sizeof(_m[0]));
	std::vector<double> a(_a, _a + sizeof(_a) / sizeof(_a[0]));
	std::vector<double> b(_b, _b + sizeof(_b) / sizeof(_b[0]));
	std::vector<double> f;
	std::string expression = "3040*((x0*x0)+(x1*x1))+1800*((x0-150)*(x0-150)+(x1-180)*(x1-180))+800*((x0-240)*(x0-240)+(x1-200)*(x1-200))+1200*((x0-260)*(x0-260)+(x1-90)*(x1-90))";
	//std::vector<double> w;
	for (size_t i = 0; i < sizeof(_f) / sizeof(_f[0]); i++) for (size_t j = 0; j <= n; j++) f.push_back(_f[i][j]);
	//for (size_t i = 0; i < sizeof(_w) / sizeof(_w[0]); i++) for (size_t j = 0; j <= n; j++) w.push_back(_w[i][j]);

	char* input_file_name = NULL;
	char* output_file_name = NULL;
	char* options_file_name = NULL;

	// Поддержка кириллицы в консоли Windows
	// Функция setlocale() имеет два параметра, первый параметр - тип категории локали, в нашем случае LC_TYPE - набор символов, второй параметр — значение локали. 
	// Вместо второго аргумента можно писать "Russian", или оставлять пустые двойные кавычки, тогда набор символов будет такой же как и в ОС.
	setlocale(LC_ALL, "");
	setlocale(LC_NUMERIC, "C");

	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-help") == 0)
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
		else if (strcmp(argv[i], "-ask") == 0) ask_mode = ASK;
		else if (strcmp(argv[i], "-noask") == 0) ask_mode = NOASK;
		else if (strcmp(argv[i], "-trace") == 0) trace_mode = TRACE;
		else if (strcmp(argv[i], "-notrace") == 0) trace_mode = NOTRACE;
		else if (strcmp(argv[i], "-n") == 0) n = atoi(argv[++i]);
		else if (strcmp(argv[i], "-e") == 0) e = atof(argv[++i]);
		else if (strcmp(argv[i], "-c") == 0) count = atoi(argv[++i]);
		else if (strcmp(argv[i], "-t") == 0) expression = string(argv[++i]);
		else if (strcmp(argv[i], "-m") == 0)
		{
			std::istringstream ss(argv[++i]);
			m.clear();
			for (size_t i = 0; i < n; i++) m.push_back(atoi(argv[++i]));
		}
		else if (strcmp(argv[i], "-a") == 0)
		{
			a.clear();
			for (size_t i = 0; i < n; i++) a.push_back(atof(argv[++i]));
		}
		else if (strcmp(argv[i], "-b") == 0)
		{
			b.clear();
			for (size_t i = 0; i < n; i++) b.push_back(atof(argv[++i]));
		}
		else if (strcmp(argv[i], "-input") == 0) input_file_name = argv[++i];
		else if (strcmp(argv[i], "-output") == 0) output_file_name = argv[++i];
		else if (strcmp(argv[i], "-options") == 0) options_file_name = argv[++i];
	}

	if (input_file_name != NULL) freopen(input_file_name, "r",stdin);
	if (output_file_name != NULL) freopen(output_file_name, "w",stdout);

	if (options_file_name != NULL)
	{
		f.clear();
		//w.clear();
		std::ifstream options(options_file_name);
		if (!options.is_open()) throw "Error opening file";
		std::string line;
		while (std::getline(options, line))
		{
			std::cout << line << std::endl;
			std::stringstream lineStream(line);
			std::string id;
			std::string cell;
			std::vector<double> x;
			std::vector<size_t> y;
			std::getline(lineStream, id, ' ');
			if (id[0] == 'N')
			{
				std::getline(lineStream, cell, ' ');
				n = stoi(cell);
			}
			if (id[0] == 'E')
			{
				std::getline(lineStream, cell, ' ');
				e = stod(cell);
			}
			if (id[0] == 'M')
			{
				while (std::getline(lineStream, cell, ' '))
				{
					y.push_back(stoi(cell));
				}
				m = y;
			}
			if (id[0] == 'A')
			{
				while (std::getline(lineStream, cell, ' '))
				{
					x.push_back(stod(cell));
				}
				a = x;
			}
			if (id[0] == 'B')
			{
				while (std::getline(lineStream, cell, ' '))
				{
					x.push_back(stod(cell));
				}
				b = x;
			}
			if (id[0] == 'F')
			{
				while (std::getline(lineStream, cell, ' '))
				{
					x.push_back(stod(cell));
				}
				for (size_t i = 0; i < x.size(); i++) f.push_back(x[i]);
			}
			//if (id[0] == 'W') for (size_t i = 0; i < x.size(); i++) w.push_back(x[i]);
			if (id[0] == 'T')
			{
				expression = line.substr(2);
			}
		}
	}

	if (ask_mode == ASK)
	{
		std::cout << "Введите размерность пространства:" << std::endl;
		std::cin >> n;
		std::cout << "Введите искомую функцию " << n << " переменных:" << std::endl;
		std::getline(std::cin, expression);

		std::cout << "Введите число сегментов по каждому из измерений m[" << n << "]:" << std::endl;
		m.clear();
		for (size_t i = 0; i < n; i++)
		{
			int x;
			std::cin >> x;
			m.push_back(x);
		}

		std::cout << "Введите минимальные координаты по каждому из измерений a[" << n << "]:" << std::endl;
		a.clear();
		for (size_t i = 0; i < n; i++)
		{
			double x;
			std::cin >> x;
			a.push_back(x);
		}

		std::cout << "Введите максимальные координаты по каждому из измерений b[" << n << "]:" << std::endl;
		b.clear();
		for (size_t i = 0; i < n; i++)
		{
			double x;
			std::cin >> x;
			b.push_back(x);
		}

		std::cout << "Введите точность вычислений:" << std::endl;
		std::cin >> e;
		std::cout << "Введите количество повторений алгоритма для замера времени:" << std::endl;
		std::cin >> count;
	}

	for (size_t i = 0; i < m.size(); i++) assert(m[i]>2);

	// Алгоритм
	clock_t time = clock();

	std::vector<unsigned> v(n);
	std::vector<double> x(n);
	std::vector<double> a1(n);
	std::vector<double> b1(n);
	double y;

	if (trace_mode == TRACE && count == 1) std::cout << "for #1" << std::endl;
	for (unsigned s = 0; s < count; s++)
	{
		std::copy(a.begin(), a.end(), a1.begin());
		std::copy(b.begin(), b.end(), b1.begin());

		if (trace_mode == TRACE && count == 1) std::cout << "while #1" << std::endl;
		while (true)
		{
			unsigned long total = total_of(m);
			int root = sqrt(total);
			unsigned long index = total;
			// Находим первую точку в области, заданной ограничениями
			for (unsigned long index1 = 0; index1 < total; index1 += root)
			{
				std::vector<bool> bools(root, false);
#pragma omp parallel for
				for (int i = 0; i < root; i++)
					if (index1 + i < total)
					{
						std::vector<unsigned> v(n);
						std::vector<double> t(n);
						vector_of(v, index1 + i, m);
						point_of(t, v, m, a1, b1);
						bools[i] = check(t, f, a1, b1);
					}
				auto it = std::find(bools.begin(), bools.end(), true);
				if (it >= bools.end()) continue;
				size_t i = std::distance(bools.begin(), it++);
				index = index1 + i;
				break;
			}

			if (index >= total)
			{
				for (size_t i = 0; i < n; i++) m[i] <<= 1u;
				continue;
			}

			vector_of(v, index, m);
			point_of(x, v, m, a1, b1);
			//y = target(x, w);
			y = calculate(expression.c_str(), x.data(), n);

			// Находим следующую точку в области, заданной ограничениями
			for (unsigned long index1 = index + 1; index1 < total; index1 += root)
			{
				std::vector<bool> bools(root, false);
				std::vector<double> doubles(root, DBL_MAX);
#pragma omp parallel for
				for (int i = 0; i < root; i++)
					if (index1 + i < total)
					{
						std::vector<unsigned> v(n);
						std::vector<double> t(n);
						vector_of(v, index1 + i, m);
						point_of(t, v, m, a1, b1);
						if (bools[i] = check(t, f, a1, b1))
						{
							//doubles[i] = target(t, w);
							doubles[i] = calculate(expression.c_str(), t.data(), n);
						}
					}
				for (auto it = std::find(bools.begin(), bools.end(), true);
				     it < bools.end();
				     it = std::find(it, bools.end(), true))
				{
					size_t i = std::distance(bools.begin(), it++);
					if (doubles[i] > y) continue;
					y = doubles[i];
					index = index1 + i;
				}
			}
			vector_of(v, index, m);
			point_of(x, v, m, a1, b1);

			if (trace_mode == TRACE && count == 1) for (size_t i = 0; i < x.size(); i++) std::cout << x[i] << " ";
			if (trace_mode == TRACE && count == 1) std::cout << "-> " << y << std::endl;

			double dd = delta(a1, b1);
			double cc = std::max(module(a1), module(b1));
			if (dd <= cc * e) break;

#pragma omp parallel for
			for (size_t k = 0; k < n; k++)
			{
				double ak = a1[k];
				double bk = b1[k];
				double xk = x[k];
				double dd = std::max(ak - bk, bk - ak);
				a1[k] = std::max(ak, xk - dd / m[k]);
				b1[k] = std::min(bk, xk + dd / m[k]);
			}
		}
	}

	time = clock() - time;
	double seconds = ((double)time) / CLOCKS_PER_SEC / count;

	std::cout << "Исполняемый файл         : " << argv[0] << std::endl;
	std::cout << "Размерность пространства : " << n << std::endl;
	std::cout << "Число сегментов          : ";
	for (size_t i = 0; i < m.size(); i++) std::cout << m[i] << " ";
	std::cout << std::endl;
	std::cout << "Минимальные координаты   : ";
	for (size_t i = 0; i < a.size(); i++) std::cout << a[i] << " ";
	std::cout << std::endl;
	std::cout << "Максимальные координаты  : ";
	for (size_t i = 0; i < b.size(); i++) std::cout << b[i] << " ";
	std::cout << std::endl;
	std::cout << "Точность вычислений      : " << e << std::endl;
	std::cout << "Точка минимума           : ";
	for (size_t i = 0; i < x.size(); i++) std::cout << x[i] << " ";
	std::cout << std::endl;
	std::cout << "Минимальное значение     : " << y << std::endl;
	std::cout << "Время вычислений (сек.)  : " << seconds << std::endl;

	getchar();
	getchar();

	return 0;
}
