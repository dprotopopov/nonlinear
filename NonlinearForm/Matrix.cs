using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace NonlinearForm
{
    /// <summary>
    ///     Темплейт класса для работы с матрицами
    /// </summary>
    /// <typeparam name="T">тип T задаётся как float, double и т.д.– в зависимости от требуемой точности вычислений</typeparam>
    public class Matrix<T> : Vector<Vector<T>>, IDisposable
    {
        public enum Search
        {
            SearchByRows = 1,
            SearchByColumns = -1,
        };

        public enum Transform
        {
            TransformByRows = 1,
            TransformByColumns = -1,
        };

        /// <summary>
        ///     Создание матрицы, состоящей из матрицы коэффициентов и вектора правой части
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        public Matrix(IEnumerable<IEnumerable<T>> a, IEnumerable<T> b)
        {
            Debug.Assert(a.Count() == b.Count());
            int count = Math.Min(a.Count(), b.Count());
            for (int i = 0; i < count; i++)
                Add(new Vector<T>(new StackListQueue<T>(a.ElementAt(i)) {b.ElementAt(i)}));
        }

        public Matrix(IEnumerable<IEnumerable<T>> a, IEnumerable<IEnumerable<T>> b)
        {
            Debug.Assert(a.Count() == b.Count());
            int count = Math.Min(a.Count(), b.Count());
            for (int i = 0; i < count; i++)
                Add(new Vector<T>(new StackListQueue<T>(a.ElementAt(i)) {b.ElementAt(i)}));
        }


        public Matrix(int rows, int cols)
        {
            for (int i = 0; i < rows; i++)
                Add(new Vector<T>(Enumerable.Repeat(default(T), cols)));
        }

        public Matrix(IEnumerable<Vector<T>> array)
        {
            AddRange(array);
        }

        public Matrix(IEnumerable<IEnumerable<T>> array)
        {
            foreach (var vector in array)
                Add(new Vector<T>(vector));
        }

        public Matrix()
        {
        }

        public Matrix(T x) : base(new Vector<T>(x))
        {
        }

        public Matrix(IEnumerable<IEnumerable<T>> matrix, IEnumerable<int> rowIds, IEnumerable<int> columnIds)
        {
            foreach (var row in rowIds.Select(matrix.ElementAt))
                Add(new Vector<T>(columnIds.Select(row.ElementAt)));
        }

        public int Rows
        {
            get { return Count; }
        }

        public int Columns
        {
            get { return this.Any() ? this.Max(row => row.Count) : 0; }
        }

        public new void Dispose()
        {
            base.Dispose();
        }

        public override string ToString()
        {
            return string.Join(Environment.NewLine, this.Select(row => row.ToString()));
        }

        public static Matrix<T> Identity(int rank)
        {
            var e = new Matrix<T>(rank, rank);
            var one = (T) (dynamic) 1;
            for (int i = 0; i < rank; i++) e[i][i] = one;
            return e;
        }

        public void AddColumn()
        {
            foreach (var row in this) row.Add(default(T));
        }

        public static Matrix<T> operator -(Matrix<T> a, Matrix<T> b)
        {
            return new Matrix<T>(a as Vector<Vector<T>> - b);
        }

        public static Matrix<T> operator +(Matrix<T> a, Matrix<T> b)
        {
            return new Matrix<T>(a as Vector<Vector<T>> + b);
        }

        public static Matrix<T> operator *(Matrix<T> a, Matrix<T> b)
        {
            Debug.Assert(a.Columns == b.Rows);
            int rows = a.Rows;
            int columns = b.Columns;
            var result = new Matrix<T>(rows, columns);
            int count = rows*columns;
            if (count <= 0) return result;
            int size1 = (rows + columns)/2;
            int size2 = (count + size1 - 1)/size1;
            Matrix<T> bb = b.Rot();
            Parallel.ForEach(Enumerable.Range(0, size1), i =>
            {
                for (int j = 0; j < size2; j++)
                {
                    int id = i*size2 + j;
                    if (id >= count) break;
                    int m = id/columns;
                    int n = id%columns;
                    result[m][n] = Vector<T>.Scalar(a[m], bb[n]);
                }
            });
            return result;
        }

        public static Matrix<T> Scalar(IEnumerable<IEnumerable<T>> a)
        {
            int rows = a.Count();
            var result = new Matrix<T>(rows, rows);
            Parallel.ForEach(Enumerable.Range(0, rows),
                m => Parallel.ForEach(Enumerable.Range(0, m),
                    n => { result[m][n] = result[n][m] = Vector<T>.Scalar(a.ElementAt(m), a.ElementAt(n)); }));
            Parallel.ForEach(Enumerable.Range(0, rows), m => { result[m][m] = Vector<T>.Square(a.ElementAt(m)); });
            return result;
        }

        public static bool IsZero(IEnumerable<IEnumerable<T>> a)
        {
            return (!a.Any()) || a.All(IsZero);
        }

        public static bool IsZero(IEnumerable<T> a)
        {
            return (!a.Any()) || a.All(IsZero);
        }

        public static bool IsZero(T arg)
        {
            return (dynamic) arg == default(T);
        }

        public void AddRow()
        {
            Add(new Vector<T>(Enumerable.Repeat(default(T), Columns)));
        }

        public void AppendColumns(IEnumerable<IEnumerable<T>> b)
        {
            Debug.Assert(Rows == b.Count());
            Debug.Assert(this.All(row => row.Count == Columns));

            int index = 0;
            foreach (var row in this)
                row.Add(b.ElementAt(index++));
        }

        public void AppendRows(IEnumerable<IEnumerable<T>> b)
        {
            foreach (var row in b)
                Add(new Vector<T>(row));
        }

        /// <summary>
        ///     Последовательно будем выбирать разрешающий элемент РЭ, который лежит на главной диагонали матрицы.
        ///     На месте разрешающего элемента получаем 1, а в самом столбце записываем нули.
        ///     Все остальные элементы матрицы, включая элементы столбца, определяются по правилу прямоугольника.
        ///     Для этого выбираем четыре числа, которые расположены в вершинах прямоугольника и всегда включают
        ///     разрешающий элемент РЭ.
        ///     НЭ = СЭ - (А*В)/РЭ
        ///     РЭ - разрешающий элемент, А и В - элементы матрицы, образующие прямоугольник с элементами СЭ и РЭ.
        /// </summary>
        public T GaussJordan(Search search = Search.SearchByRows,
            Transform transform = Transform.TransformByRows,
            int first = 0,
            int last = Int32.MaxValue)
        {
            int row = Math.Min(Rows, last);
            int col = Math.Min(Columns, last);
            dynamic d = (T) (dynamic) 1;

            for (int i = first;
                i < Math.Min(Math.Min(Rows, Columns), last) && FindNotZero(search, i, ref row, ref col);
                i++)
            {
                d *= GaussJordanStep(transform, row, col);
                row = Math.Min(Rows, last);
                col = Math.Min(Columns, last);
            }
            return d;
        }

        private bool FindNotZero(Search search, int i, ref int row, ref int col)
        {
            Debug.Assert(row <= Rows);
            Debug.Assert(col <= Columns);
            switch (search)
            {
                case Search.SearchByRows:
                    for (int j = 0, total = (row - i)*col, n = col; j < total; j++)
                    {
                        row = i + (j/n);
                        col = (j%n);
                        if (!IsZero(this[row][col])) return true;
                    }
                    return false;
                case Search.SearchByColumns:
                    for (int j = 0, total = row*(col - i), n = row; j < total; j++)
                    {
                        col = i + (j/n);
                        row = (j%n);
                        if (!IsZero(this[row][col])) return true;
                    }
                    return false;
            }
            throw new NotImplementedException();
        }

        public T GaussJordanStep(Transform transform, int row, int col)
        {
            dynamic d = this[row][col];

            var rowIds = new StackListQueue<int>();
            var columnIds = new StackListQueue<int>();

            for (int i = 0; i < Rows; i++) if (!IsZero(this[i][col])) rowIds.Add(i);
            for (int j = 0; j < Columns; j++) if (!IsZero(this[row][j])) columnIds.Add(j);

            rowIds.Remove(row);
            columnIds.Remove(col);

            int rows = rowIds.Count();
            int columns = columnIds.Count();
            int count = rows*columns;
            if (count > 0)
            {
                int size1 = (rows + columns)/2;
                int size2 = (count + size1 - 1)/size1;

                Parallel.ForEach(Enumerable.Range(0, size1), i =>
                {
                    for (int j = 0; j < size2; j++)
                    {
                        int id = i*size2 + j;
                        if (id >= count) break;
                        int m = rowIds[id/columns];
                        int n = columnIds[id%columns];
                        dynamic a = this[m][n];
                        dynamic b = this[m][col];
                        dynamic c = this[row][n];
                        this[m][n] = a - b*c/d;
                    }
                });
            }
            switch (transform)
            {
                case Transform.TransformByRows:
                    Parallel.ForEach(rowIds, i => { this[i][col] = default(T); });
                    Parallel.ForEach(columnIds, j =>
                    {
                        dynamic a = this[row][j];
                        this[row][j] = a/d;
                    });
                    break;
                case Transform.TransformByColumns:
                    Parallel.ForEach(columnIds, j => { this[row][j] = default(T); });
                    Parallel.ForEach(rowIds, i =>
                    {
                        dynamic a = this[i][col];
                        this[i][col] = a/d;
                    });
                    break;
            }

            this[row][col] = (T) (dynamic) 1;

            return d;
        }

        /// <summary>
        ///     Определитель матрицы
        ///     Матрица обратима тогда и только тогда,
        ///     когда определитель матрицы отличен от нуля
        /// </summary>
        public T Det()
        {
            Debug.Assert(Rows == Columns);
            // Приведение матрицы к каноническому виду
            dynamic d = GaussJordan();
            // Проверка на нулевые строки
            if (this.Any(IsZero)) return default(T);
            int[] array = this.Select(row => row.IndexOf(row.First(Vector<T>.NotZero))).ToArray();
            int rows = Rows;
            long parity = 0;
            for (int i = 0; i < rows; i++)
                for (int j = i + 1; j < rows; j++)
                {
                    int a = array[i];
                    int b = array[j];
                    if (a <= b) continue;
                    array[i] = b;
                    array[j] = a;
                    parity++;
                }
            return ((parity & 1) == 0) ? d : (- d);
        }

        /// <summary>
        ///     Вычисление обратной матрицы
        /// </summary>
        /// <returns></returns>
        public Matrix<T> Inv()
        {
            Debug.Assert(Rows == Columns);
            var ab = new Matrix<T>(this, Identity(Rows));

            ab.GaussJordan();
            if (new Matrix<T>(ab.Select(row => row.GetRange(0, Rows))).Any(Vector<T>.IsZero))
                throw new DivideByZeroException();

            var index = new Vector<int>(ab.Select(row => row.IndexOf(row.First(Vector<T>.NotZero))));
            int count = index.Count;
            var index2 = new Vector<int>(Enumerable.Range(0, count));
            for (int i = 0; i < count; i++) index2[index[i]] = i;
            return new Matrix<T>(index2.Select(item => ab[item].GetRange(Rows, Rows)));
        }

        /// <summary>
        ///     Транспонирование матрицы
        /// </summary>
        /// <returns></returns>
        public Matrix<T> Rot()
        {
            var matrix = new Matrix<T>();
            for (int i = 0; i < Columns; i++) matrix.Add(new Vector<T>());
            foreach (var row in this)
            {
                int index = 0;
                foreach (T item in row)
                    matrix[index++].Add(item);
            }
            return matrix;
        }
    }
}