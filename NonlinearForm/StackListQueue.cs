using System;
using System.Collections.Generic;
using System.Linq;

namespace NonlinearForm
{
    /// <summary>
    ///     Класс реализующий интерфейсы для работы с коллекцией данных как списком, как стеком, как с очёредью
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class StackListQueue<T> : List<T>
    {
        #region

        public StackListQueue(IEnumerable<T> value)
        {
            base.AddRange(value);
        }

        public StackListQueue(T value)
        {
            base.Add(value);
        }

        public StackListQueue()
        {
        }

        #endregion


        public void Enqueue(T value)
        {
            base.Add(value);
        }

        public void Enqueue(IEnumerable<T> value)
        {
            if (!value.Any()) return;
            base.AddRange(value);
        }

        public void ReplaceAll(IEnumerable<T> value)
        {
            Clear();
            if (!value.Any()) return;
            base.AddRange(value);
        }

        public T Dequeue()
        {
            T value = this[0];
            RemoveAt(0);
            return value;
        }

        public void Rotate()
        {
            base.Add(this[0]);
            base.RemoveAt(0);
        }

        public void Rotate(int count)
        {
            base.AddRange(GetRange(0, count));
            base.RemoveRange(0, count);
        }

        public IEnumerable<T> Dequeue(int count)
        {
            IEnumerable<T> value = GetRange(0, count);
            RemoveRange(0, count);
            return value;
        }

        public IEnumerable<T> GetReverse()
        {
            int count = Count - 1;
            return this.Select((t, i) => this[count - i]);
        }

        public void Push(T value)
        {
            base.Add(value);
        }

        public void Prepend(T value)
        {
            Insert(0, value);
        }

        public void Add(IEnumerable<T> value)
        {
            if (!value.Any()) return;
            base.AddRange(value);
        }

        public virtual void AddExcept(T item)
        {
            if (!Contains(item)) Add(item);
        }

        public virtual void AddRangeExcept(IEnumerable<T> value)
        {
            if (!value.Any()) return;
            base.AddRange(value.Except(this));
        }

        public T Pop()
        {
            int index = Count;
            T value = this[--index];
            RemoveAt(index);
            return value;
        }

        public IEnumerable<T> Pop(int count)
        {
            int index = Count - count;
            IEnumerable<T> value = GetRange(index, count);
            RemoveRange(index, count);
            return value;
        }

        public void Push(IEnumerable<T> value)
        {
            if (!value.Any()) return;
            base.AddRange(value);
        }

        public virtual bool Contains(IEnumerable<T> collection)
        {
            return collection.All(Contains);
        }

        public virtual bool BelongsTo(IEnumerable<T> collection)
        {
            if (!this.All(collection.Contains)) return false;
            var forward = new StackListQueue<T>(collection);
            for (int index = forward.IndexOf(this[0]);
                index >= 0 && index + Count <= forward.Count();
                index = forward.IndexOf(this[0]))
            {
                if (this.SequenceEqual(forward.GetRange(index, Count))) return true;
                forward.RemoveRange(0, index);
            }
            return false;
        }

        public virtual StackListQueue<int> GetInts(T values)
        {
            throw new NotImplementedException();
        }

        public override int GetHashCode()
        {
            return this.Aggregate(0,
                (current, item) => (current << 1) ^ (current >> (8*sizeof (int) - 1)) ^ item.GetHashCode());
        }

        public override string ToString()
        {
            return string.Join(",", this.Select(item => item.ToString()));
        }

        public void Append(T value)
        {
            base.Add(value);
        }
    }
}