using System;
using System.Drawing;
using System.Globalization;
using System.Windows.Forms;

namespace NonlinearForm.Forms
{
    public partial class MatrixControl : UserControl
    {
        public MatrixControl()
        {
            InitializeComponent();
            //used to attach event-handlers to the events of the editing control(nice name!)
        }

        public int Rows
        {
            get { return dataGridView1.RowCount; }
            set
            {
                while (dataGridView1.RowCount < value)
                    dataGridView1.Rows.Add(new DataGridViewRow
                    {
                        HeaderCell =
                        {
                            Value = dataGridView1.RowCount.ToString(CultureInfo.InvariantCulture)
                        }
                    });
                while (dataGridView1.RowCount > value)
                    dataGridView1.Rows.RemoveAt(dataGridView1.RowCount - 1);
            }
        }

        public bool ReadOnly
        {
            get { return dataGridView1.ReadOnly; }
            set { dataGridView1.ReadOnly = value; }
        }

        public int Columns
        {
            get { return dataGridView1.Columns.Count; }
            set
            {
                while (dataGridView1.Columns.Count < value)
                    dataGridView1.Columns.Add(new DataGridViewTextBoxColumn
                    {
                        HeaderText = dataGridView1.Columns.Count.ToString(CultureInfo.InvariantCulture),
                        Name = "Column" + dataGridView1.Columns.Count,
                        Width = 60,
                        SortMode = DataGridViewColumnSortMode.NotSortable,
                        ValueType = typeof (string),
                        CellTemplate = new DataGridViewTextBoxCell
                        {
                            ValueType = typeof (string),
                            Style =
                            {
                                BackColor = Color.LightCyan,
                                SelectionBackColor = Color.FromArgb(128, 255, 255),
                                Alignment = DataGridViewContentAlignment.MiddleRight
                            }
                        }
                    });
                while (dataGridView1.Columns.Count > value)
                    dataGridView1.Columns.RemoveAt(dataGridView1.Columns.Count - 1);
            }
        }

        public Matrix<decimal> Value
        {
            get
            {
                var rowCount = dataGridView1.RowCount;
                var columnCount = dataGridView1.ColumnCount;
                var rows = new Vector<Vector<decimal>>();
                for (var r = 0; r < rowCount; r++)
                {
                    var columns = new Vector<decimal>();
                    for (var c = 0; c < columnCount; c++)
                        columns.Add(Convert.ToDecimal(dataGridView1.Rows[r].Cells[c].Value));
                    rows.Add(columns);
                }
                return new Matrix<decimal>(rows);
            }
            set
            {
                var columnCount = Columns = value.Columns;
                var rowCount = Rows = value.Rows;
                for (var r = 0; r < rowCount; r++)
                    for (var c = 0; c < columnCount; c++)
                        dataGridView1.Rows[r].Cells[c].Value = value[r][c].ToString(CultureInfo.InvariantCulture);
            }
        }
    }
}