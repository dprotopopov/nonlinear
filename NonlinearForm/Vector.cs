using System;
using System.Collections.Generic;
using System.Linq;

namespace NonlinearForm
{
    /// <summary>
    ///     Темплейт класса для работы с векторами
    /// </summary>
    /// <typeparam name="T">тип T задаётся как float, double и т.д.– в зависимости от требуемой точности вычислений</typeparam>
    public class Vector<T> : List<T>, IDisposable
    {
        public Vector()
        {
        }

        public Vector(int n, Random random)
        {
            for (int i = 0; i < n; i++) Add((T) (dynamic) random.NextDouble());
        }

        public Vector(T value)
        {
            Add(value);
        }

        public Vector(IEnumerable<T> vector)
            : base(vector)
        {
        }

        public override string ToString()
        {
            return string.Join(";", this.Select(item => item.ToString()));
        }

        public void Dispose()
        {
        }

        public static Vector<T> operator +(Vector<T> a, Vector<T> b)
        {
            var c = new Vector<T>(a.Zip(b, (first, second) => (T) ((dynamic) first + second)));
            if (a.Count > b.Count) c.Add(a.GetRange(b.Count, a.Count - b.Count));
            if (a.Count < b.Count) c.Add(b.GetRange(a.Count, b.Count - a.Count));
            return c;
        }

        public static Vector<T> operator -(Vector<T> a, Vector<T> b)
        {
            var c = new Vector<T>(a.Zip(b, (first, second) => (T) ((dynamic) first - second)));
            if (a.Count > b.Count) c.Add(a.GetRange(b.Count, a.Count - b.Count));
            if (a.Count < b.Count) c.Add(b.GetRange(a.Count, b.Count - a.Count).Select(item => (T) (-(dynamic) item)));
            return c;
        }

        public static Vector<T> operator -(Vector<T> a)
        {
            return new Vector<T>(a.Select(t => (T) (-(dynamic) t)));
        }

        public static bool IsZero(IEnumerable<T> a)
        {
            return (!a.Any()) || a.All(IsZero);
        }

        public static bool IsZero(T arg)
        {
            return (dynamic) arg == default(T);
        }

        public static bool NotZero(T arg)
        {
            return (dynamic) arg != default(T);
        }

        public static T Scalar(IEnumerable<T> a, IEnumerable<T> b)
        {
            return a.Zip(b, (first, second) => ((dynamic) first*second)).Aggregate(default(T), (x, y) => x + y);
        }

        public static T Square(IEnumerable<T> a)
        {
            return a.Select(item => ((dynamic) item*item)).Aggregate(default(T), (x, y) => x + y);
        }

        public void Add(IEnumerable<T> value)
        {
            if (!value.Any()) return;
            base.AddRange(value);
        }

        public static T Sum(IEnumerable<T> a)
        {
            return a.Aggregate(default(T), (x, y) => (dynamic) x + y);
        }
    }
}