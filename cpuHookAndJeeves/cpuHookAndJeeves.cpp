// �������� ���� � ������ � �������������� ���������� �����������
// ������ �., ����� �.
// ���������� ����������������. ������ � ���������:
// ���. � ����. - �.: ���, 1982.
// 583 �.

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <numeric> // std::accumulate


#include <locale>
#include <assert.h>
#include <fstream>

using namespace std;

double module(std::vector<double>& x);
double delta(std::vector<double>& x, std::vector<double>& y);
double distance(std::vector<double>& x, std::vector<double>& y);
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
// ��������� ��������
static const unsigned _count = 1;
static const size_t _n = 2;
static const size_t _md = 20;
static const size_t _m[] = {20, 20};
static const double _a[] = {0, 0};
static const double _b[] = {1000, 1000};
static const double _f1[] = {0, 0, 500};
static const double _f2[] = {100, 100, 500};
static const double* _f[] = {_f1, _f2};
static const double _w1[] = {0, 0, 3040};
static const double _w2[] = {150, 180, 1800};
static const double _w3[] = {240, 200, 800};
static const double _w4[] = {260, 90, 1200};
static const double* _w[] = {_w1, _w2, _w3, _w4};
static const double _e = 1e-8;


/////////////////////////////////////////////////////////
// ���������� ������ �������
double module(std::vector<double>& x)
{
	std::vector<double> y(x.size());
	std::transform(x.begin(), x.end(), y.begin(), abs_functor<double>);
	return std::accumulate(y.begin(), y.end(), 0.0, max_functor<double>);
}

/////////////////////////////////////////////////////////
// ���������� ��������� ����� ����� ��������� ���������
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
// ���������� ��������� ����� ����� ��������� ���������
double distance(std::vector<double>& x, std::vector<double>& y)
{
	size_t i = std::min(x.size(), y.size());
	std::vector<double> sub(std::max(x.size(), y.size()));
	std::vector<double> square(std::max(x.size(), y.size()));
	std::transform(x.begin(), x.end(), y.begin(), sub.begin(), sub_functor<double>);
	std::copy(x.begin() + i, x.end(), sub.begin() + i);
	std::copy(y.begin() + i, y.end(), sub.begin() + i);
	std::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>);
	return std::sqrt(std::accumulate(square.begin(), square.end(), 0.0, add_functor<double>));
}

/////////////////////////////////////////////////////////
// ���������� ������� �������� ��������� ������� �� ������ ����
// vector - ������ �������� ��������� �������
// index - ����� ���� �������
// m - ����� ��������� �� ������� �� ���������
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
// �������������� ������� �������� ��������� �������
// � ������ ��������� �����
// vector - ������ �������� ��������� �������
// m - ����� ��������� �� ������� �� ���������
// a - ������ ����������� ��������� �����
// b - ������ ������������ ��������� �����
void point_of(std::vector<double>& point, std::vector<unsigned>& vector, std::vector<size_t>& m, std::vector<double>& a, std::vector<double>& b)
{
	for (size_t i = 0; i < m.size(); i++) point[i] = (a[i] * (m[i] - vector[i]) + b[i] * vector[i]) / m[i];
}

/////////////////////////////////////////////////////////
// ���������� ����� ����� �������
// m - ����� ��������� �� ������� �� ���������
unsigned long total_of(std::vector<size_t>& m)
{
	std::vector<size_t> m1(m.size());
	std::transform(m.begin(), m.end(), m1.begin(), inc_functor<size_t>);
	return std::accumulate(m1.begin(), m1.end(), 1UL, mul_functor<unsigned long>);
}

/////////////////////////////////////////////////////////
// �������� �������������� ����� �������, �������� �������������
// x - ���������� �����
// f - ����� ����������� �������
// a - ������ ����������� ��������� �����
// b - ������ ������������ ��������� �����
bool check(std::vector<double>& x, std::vector<double>& f, std::vector<double>& a, std::vector<double>& b)
{
	size_t n = x.size();
	for (size_t i = 0; i < n; i++) if (x[i] < a[i] && x[i] < b[i]) return false;
	for (size_t i = 0; i < n; i++) if (x[i] > a[i] && x[i] > b[i]) return false;
	for (size_t i = 0; i < f.size() / (n+1); i++)
	{
		std::vector<double> sub(x.size());
		std::vector<double> square(x.size());
		std::transform(x.begin(), x.end(), f.begin() + i * (n + 1), sub.begin(), diff_functor<double>);
		std::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>);
		if (std::sqrt(std::accumulate(square.begin(), square.end(), 0.0, add_functor<double>)) > f[i * (n + 1) + n]) return false;
	}
	return true;
}

/////////////////////////////////////////////////////////
// ������� �������
double target(std::vector<double>& x, std::vector<double>& w)
{
	size_t n = x.size();
	double s = 0;
	for (size_t i = 0; i < w.size() / (n+1); i++)
	{
		std::vector<double> sub(x.size());
		std::vector<double> square(x.size());
		std::transform(x.begin(), x.end(), w.begin() + i * (n + 1), sub.begin(), diff_functor<double>);
		std::transform(sub.begin(), sub.end(), square.begin(), square_functor<double>);
		s += std::sqrt(std::accumulate(square.begin(), square.end(), 0.0, add_functor<double>)) * w[i * (n + 1) + n];
	}
	return s;
}

int main(int argc, char* argv[])
{
	// http://stackoverflow.com/questions/2236197/what-is-the-easiest-way-to-initialize-a-stdvector-with-hardcoded-elements

	unsigned count = _count;
	size_t n = _n;
	double e = _e;
	size_t md = _md;
	std::vector<size_t> m(_m, _m + sizeof(_m) / sizeof(_m[0]));
	std::vector<double> a(_a, _a + sizeof(_a) / sizeof(_a[0]));
	std::vector<double> b(_b, _b + sizeof(_b) / sizeof(_b[0]));
	std::vector<double> f;
	std::vector<double> w;
	for (size_t i = 0; i < sizeof(_f) / sizeof(_f[0]); i++) for (size_t j = 0; j <= n; j++) f.push_back(_f[i][j]);
	for (size_t i = 0; i < sizeof(_w) / sizeof(_w[0]); i++) for (size_t j = 0; j <= n; j++) w.push_back(_w[i][j]);

	char* input_file_name = NULL;
	char* output_file_name = NULL;
	char* options_file_name = NULL;

	// ��������� ��������� � ������� Windows
	// ������� setlocale() ����� ��� ���������, ������ �������� - ��� ��������� ������, � ����� ������ LC_TYPE - ����� ��������, ������ �������� � �������� ������. 
	// ������ ������� ��������� ����� ������ "Russian", ��� ��������� ������ ������� �������, ����� ����� �������� ����� ����� �� ��� � � ��.
	setlocale(LC_ALL, "");

	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-help") == 0)
		{
			std::cout << "Usage :\t" << argv[0] << " [...] [-input <inputfile>] [-output <outputfile>]" << std::endl;
			std::cout << "�������� ���� � ������ � �������������� ���������� �����������" << std::endl;
			std::cout << "(�������� ������� �������� ��������� �������)" << std::endl;
			//			std::cout << "\t-n <����������� ������������>" << std::endl;
			std::cout << "\t-c <���������� ���������� ��������� ��� ������ �������>" << std::endl;
			std::cout << "\t-m <����� ��������� �� ������� �� ���������>" << std::endl;
			std::cout << "\t-md <����� ��������� �� ������������ �����������>" << std::endl;
			std::cout << "\t-a <����������� ���������� �� ������� �� ���������>" << std::endl;
			std::cout << "\t-b <������������ ���������� �� ������� �� ���������>" << std::endl;
			std::cout << "\t-e <�������� ����������>" << std::endl;
			std::cout << "\t-ask/noask" << std::endl;
			std::cout << "\t-trace/notrace" << std::endl;
		}
		else if (strcmp(argv[i], "-ask") == 0) ask_mode = ASK;
		else if (strcmp(argv[i], "-noask") == 0) ask_mode = NOASK;
		else if (strcmp(argv[i], "-trace") == 0) trace_mode = TRACE;
		else if (strcmp(argv[i], "-notrace") == 0) trace_mode = NOTRACE;
		//		else if(strcmp(argv[i],"-n")==0) n = atoi(argv[++i]);
		else if (strcmp(argv[i], "-e") == 0) e = atof(argv[++i]);
		else if (strcmp(argv[i], "-c") == 0) count = atoi(argv[++i]);
		else if (strcmp(argv[i], "-md") == 0) md = atoi(argv[++i]);
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
		w.clear();
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
			while (std::getline(lineStream, cell, ' '))
			{
				x.push_back(stod(cell));
				y.push_back(stoi(cell));
			}
			if (id[0] == 'N') n = stoi(cell);
			if (id[0] == 'D') md = stoi(cell);
			if (id[0] == 'E') e = stod(cell);
			if (id[0] == 'M') m = y;
			if (id[0] == 'A') a = x;
			if (id[0] == 'B') b = x;
			if (id[0] == 'F') for (size_t i = 0; i < x.size(); i++) f.push_back(x[i]);
			if (id[0] == 'W') for (size_t i = 0; i < x.size(); i++) w.push_back(x[i]);
		}
	}

	if (ask_mode == ASK)
	{
		//  std::cout << "������� ����������� ������������:"<< std::endl; std::cin >> n;

		std::cout << "������� ����� ��������� �� ������� �� ��������� m[" << n << "]:" << std::endl;
		m.clear();
		for (size_t i = 0; i < n; i++)
		{
			int x;
			std::cin >> x;
			m.push_back(x);
		}

		std::cout << "������� ����� ��������� �� ������������ �����������:" << std::endl;
		std::cin >> md;

		std::cout << "������� ����������� ���������� �� ������� �� ��������� a[" << n << "]:" << std::endl;
		a.clear();
		for (size_t i = 0; i < n; i++)
		{
			double x;
			std::cin >> x;
			a.push_back(x);
		}

		std::cout << "������� ������������ ���������� �� ������� �� ��������� b[" << n << "]:" << std::endl;
		b.clear();
		for (size_t i = 0; i < n; i++)
		{
			double x;
			std::cin >> x;
			b.push_back(x);
		}

		std::cout << "������� �������� ����������:" << std::endl;
		std::cin >> e;
		std::cout << "������� ���������� ���������� ��������� ��� ������ �������:" << std::endl;
		std::cin >> count;
	}

	for (size_t i = 0; i < m.size(); i++) assert(m[i]>2);
	assert(md>2);

	// ��������
	clock_t time = clock();

	double diameter = ::distance(a, b);

	std::vector<unsigned> v(n);
	std::vector<double> t(n);
	std::vector<double> x(n);
	std::vector<double> x1(n);
	std::vector<double> x2(n);
	double y;

	if (trace_mode == TRACE && count == 1) std::cout << "for #1" << std::endl;
	for (unsigned s = 0; s < count; s++)
	{
		if (trace_mode == TRACE && count == 1) std::cout << "while #1" << std::endl;
		while (true)
		{
			// ������� ������ ����� � �������, �������� �������������
			unsigned long total = total_of(m);
			unsigned long index = 0;
			while (index < total)
			{
				vector_of(v, index++, m);
				point_of(x, v, m, a, b);
				if (check(x, f, a, b)) break;
			}
			if (index >= total)
			{
				for (size_t i = 0; i < n; i++) m[i] <<= 1u;
				continue;
			}
			y = target(x, w);
			if (trace_mode == TRACE && count == 1) for (size_t i = 0; i < x.size(); i++) std::cout << x[i] << " ";
			if (trace_mode == TRACE && count == 1) std::cout << "-> " << y << std::endl;
			break;
		}

		if (trace_mode == TRACE && count == 1) std::cout << "while #2" << std::endl;
		while (true)
		{
			// ������� ��������� ����� � �������, �������� �������������
			// ��������� �������� ���������� ����������� �� �����������
			// ������������� �������� ��� ������������

			std::copy(x.begin(), x.end(), x1.begin()); // ���������� �������� ��������� �����

			// ���� �� ����������
			if (trace_mode == TRACE && count == 1) std::cout << "for #2" << std::endl;
			for (size_t k = 0; k < n; k++)
			{
				// �������� ���������� ����������� �� �����������
				double ak = std::min(a[k], b[k]);
				double bk = std::max(a[k], b[k]);
				size_t mk = m[k];
				std::copy(x.begin(), x.end(), t.begin());
				while (true)
				{
					for (size_t index = 0; index <= mk; index++)
					{
						t[k] = (ak * (mk - index) + bk * index) / mk;
						if (!check(t, f, a, b)) continue;
						double yk = target(t, w);
						if (yk > y) continue;
						y = yk;
						x[k] = t[k];
						if (trace_mode == TRACE && count == 1) for (size_t i = 0; i < x.size(); i++) std::cout << x[i] << " ";
						if (trace_mode == TRACE && count == 1) std::cout << "-> " << y << std::endl;
					}
					double dd = std::max(ak - bk, bk - ak);
					double cc = std::max(std::max(ak, -ak), std::max(-bk, bk));
					if (dd <= cc * e) break;
					double xk = x[k];
					ak = std::max(ak, xk - dd / mk);
					bk = std::min(bk, xk + dd / mk);
				}
			}

			std::copy(x.begin(), x.end(), x2.begin()); // ���������� �������� ��������� �����

			// ������� ��������� ����� � �������, �������� �������������
			// ��������� �������� ���������� ����������� �� ����������� x2->x1
			double l = -diameter;
			double h = diameter;

			if (trace_mode == TRACE && count == 1) std::cout << "while #3" << std::endl;
			while (true)
			{
				double p = 0;

				for (size_t index = 0; index <= md + md; index++)
				{
					double pt = (l * (md + md - index) + h * index) / (md + md);
					for (size_t i = 0; i < n; i++) t[i] = x2[i] * (1.0 - pt) + x1[i] * pt;
					if (!check(t, f, a, b)) continue;
					double yt = target(t, w);
					if (yt > y) continue;
					p = pt;
					y = yt;
					std::copy(t.begin(), t.end(), x.begin());
					if (trace_mode == TRACE && count == 1) for (size_t i = 0; i < x.size(); i++) std::cout << x[i] << " ";
					if (trace_mode == TRACE && count == 1) std::cout << "-> " << y << std::endl;
				}
				double dd = std::max(h - l, l - h);
				double cc = std::max(std::max(h, -h), std::max(-l, l));
				if (dd <= cc * e) break;
				double ll = l;
				double hh = h;
				l = std::max(ll, p - dd / md);
				h = std::min(hh, p + dd / md);
			}

			double dd = delta(x, x1);
			double cc = std::max(module(x), module(x1));
			if (dd <= cc * e) break;
		}
	}

	time = clock() - time;
	double seconds = ((double)time) / CLOCKS_PER_SEC / count;

	std::cout << "����������� ����         : " << argv[0] << std::endl;
	std::cout << "����������� ������������ : " << n << std::endl;
	std::cout << "����� ���������          : ";
	for (size_t i = 0; i < m.size(); i++) std::cout << m[i] << " ";
	std::cout << "+ " << md;
	std::cout << std::endl;
	std::cout << "����������� ����������   : ";
	for (size_t i = 0; i < a.size(); i++) std::cout << a[i] << " ";
	std::cout << std::endl;
	std::cout << "������������ ����������  : ";
	for (size_t i = 0; i < b.size(); i++) std::cout << b[i] << " ";
	std::cout << std::endl;
	std::cout << "�������� ����������      : " << e << std::endl;
	std::cout << "����� ��������           : ";
	for (size_t i = 0; i < x.size(); i++) std::cout << x[i] << " ";
	std::cout << std::endl;
	std::cout << "����������� ��������     : " << y << std::endl;
	std::cout << "����� ���������� (���.)  : " << seconds << std::endl;

	getchar();
	getchar();

	return 0;
}
