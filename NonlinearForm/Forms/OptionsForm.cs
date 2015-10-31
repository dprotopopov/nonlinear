using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace NonlinearForm.Forms
{
    public partial class OptionsForm : Form
    {
        public OptionsForm()
        {
            InitializeComponent();
            try
            {
                using (var reader = File.OpenText("math.default"))
                {
                    var n = 0;
                    var m = new Matrix<decimal>();
                    var a = new Matrix<decimal>();
                    var b = new Matrix<decimal>();
                    var f = new Matrix<decimal>();
                    var w = new Matrix<decimal>();
                    for (var line = reader.ReadLine(); !string.IsNullOrWhiteSpace(line); line = reader.ReadLine())
                    {
                        var arr = line.Split(' ');
                        if (arr[0][0] == 'N')
                        {
                            n = Convert.ToInt32(arr[1]);
                        }
                        else
                        {
                            var x = new Vector<decimal>();
                            for (var i = 1; i < arr.Length; i++) x.Add(Convert.ToDecimal(arr[i]));
                            if (arr[0][0] == 'M') m.Add(x);
                            if (arr[0][0] == 'A') a = new Matrix<decimal> {x};
                            if (arr[0][0] == 'B') b = new Matrix<decimal> {x};
                            if (arr[0][0] == 'F') f.Add(x);
                            if (arr[0][0] == 'W') w.Add(x);
                        }
                    }
                    numericUpDown1.Value = n;
                    _matrixControlM.Value = m;
                    _matrixControlA.Value = a;
                    _matrixControlB.Value = b;
                    _matrixControlF.Value = f;
                    _matrixControlW.Value = w;
                }
            }
            catch (Exception ex)
            {
            }
        }

        private void numericUpDown1_ValueChanged(object sender, EventArgs e)
        {
            var n = (int) numericUpDown1.Value;
            _matrixControlM.Columns = n;
            _matrixControlA.Columns = n;
            _matrixControlB.Columns = n;
            _matrixControlF.Columns = n + 1;
            _matrixControlW.Columns = n + 1;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            _matrixControlF.Rows += 1;
        }

        private void button4_Click(object sender, EventArgs e)
        {
            _matrixControlW.Rows += 1;
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (_matrixControlF.Rows > 0) _matrixControlF.Rows -= 1;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            if (_matrixControlW.Rows > 0) _matrixControlW.Rows -= 1;
        }

        private void button5_Click(object sender, EventArgs e)
        {
            if (openFileDialog1.ShowDialog() != DialogResult.OK) return;
            var fileName = openFileDialog1.FileName;
            try
            {
                using (var reader = File.OpenText(fileName))
                {
                    var n = 0;
                    var m = new Matrix<decimal>();
                    var a = new Matrix<decimal>();
                    var b = new Matrix<decimal>();
                    var f = new Matrix<decimal>();
                    var w = new Matrix<decimal>();
                    for (var line = reader.ReadLine(); !string.IsNullOrWhiteSpace(line); line = reader.ReadLine())
                    {
                        var arr = line.Split(' ');
                        if (arr[0][0] == 'N')
                        {
                            n = Convert.ToInt32(arr[1]);
                        }
                        else
                        {
                            var x = new Vector<decimal>();
                            for (var i = 1; i < arr.Length; i++) x.Add(Convert.ToDecimal(arr[i]));
                            if (arr[0][0] == 'M') m.Add(x);
                            if (arr[0][0] == 'A') a = new Matrix<decimal> {x};
                            if (arr[0][0] == 'B') b = new Matrix<decimal> {x};
                            if (arr[0][0] == 'F') f.Add(x);
                            if (arr[0][0] == 'W') w.Add(x);
                        }
                    }
                    numericUpDown1.Value = n;
                    _matrixControlM.Value = m;
                    _matrixControlA.Value = a;
                    _matrixControlB.Value = b;
                    _matrixControlF.Value = f;
                    _matrixControlW.Value = w;
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void button6_Click(object sender, EventArgs e)
        {
            if (saveFileDialog1.ShowDialog() != DialogResult.OK) return;
            var fileName = saveFileDialog1.FileName;
            try
            {
                using (var writer = File.CreateText(fileName))
                {
                    var n = (int) numericUpDown1.Value;
                    var m = _matrixControlM.Value;
                    var a = _matrixControlA.Value;
                    var b = _matrixControlB.Value;
                    var f = _matrixControlF.Value;
                    var w = _matrixControlW.Value;
                    writer.WriteLine("N {0}", n);
                    foreach (var row in m)
                        writer.WriteLine("M {0}",
                            string.Join(" ", row.Select(x => x.ToString(CultureInfo.InvariantCulture))));
                    foreach (var row in a)
                        writer.WriteLine("A {0}",
                            string.Join(" ", row.Select(x => x.ToString(CultureInfo.InvariantCulture))));
                    foreach (var row in b)
                        writer.WriteLine("B {0}",
                            string.Join(" ", row.Select(x => x.ToString(CultureInfo.InvariantCulture))));
                    foreach (var row in f)
                        writer.WriteLine("F {0}",
                            string.Join(" ", row.Select(x => x.ToString(CultureInfo.InvariantCulture))));
                    foreach (var row in w)
                        writer.WriteLine("W {0}",
                            string.Join(" ", row.Select(x => x.ToString(CultureInfo.InvariantCulture))));
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

    }
}