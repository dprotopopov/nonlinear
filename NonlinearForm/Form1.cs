using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NonlinearForm.Forms;

namespace NonlinearForm
{
    public partial class Form1 : Form
    {
        OptionsForm form = new OptionsForm();
        public Form1()
        {
            InitializeComponent();
        }

        private void выходToolStripMenuItem_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        private void cpudichotomy_Click(object sender, EventArgs e)
        {
            StartProcessCPU("cpudichotomy");
        }

        void StartProcessGPU(string alg)
        {
            File.Delete("result.txt");
            string cmd = string.Format("/C {0} g {1} b {2} >> result.txt", alg, (int)numericUpDown1.Value, (int)numericUpDown2.Value);
            Process process = new Process();
            ProcessStartInfo info = new ProcessStartInfo();
            //info.WindowStyle = ProcessWindowStyle.Hidden;
            info.FileName = "cmd";
            info.Arguments = cmd;
            process.StartInfo = info;
            process.Start();
            process.WaitForExit();
            MessageBox.Show("Выполнено.", "", MessageBoxButtons.OK, MessageBoxIcon.Information);
            if (!process.HasExited)
                process.Kill();
            richTextBox1.Text += File.ReadAllText(@"result.txt", Encoding.Default) + "\n\n";
        }

        void StartProcessCPU(string alg)
        {
            File.Delete("result.txt");
            string cmd = string.Format("/C {0} >> result.txt", alg);
            Process process = new Process();
            ProcessStartInfo info = new ProcessStartInfo();
            //info.WindowStyle = ProcessWindowStyle.Hidden;
            info.FileName = "cmd";
            info.Arguments = cmd;
            process.StartInfo = info;
            process.Start();
            process.WaitForExit();
            MessageBox.Show("Выполнено.", "", MessageBoxButtons.OK, MessageBoxIcon.Information);
            if (!process.HasExited)
                process.Kill();
            richTextBox1.Text += File.ReadAllText(@"result.txt", Encoding.Default)+"\n\n";
        }

        private void clean_Click(object sender, EventArgs e)
        {
            File.Delete(@"result.txt");
            richTextBox1.Clear();
            MessageBox.Show("Выполнено.", "", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void cpucyclic_Click(object sender, EventArgs e)
        {
            StartProcessCPU("cpucyclic");
        }

        private void cpuHookAndJeeves_Click(object sender, EventArgs e)
        {
            StartProcessCPU("cpuHookAndJeeves");
        }

        private void gpudichotomy_Click(object sender, EventArgs e)
        {
            StartProcessGPU("cudadichotomy2");
        }

        private void gpucyclic_Click(object sender, EventArgs e)
        {
            StartProcessGPU("cudacyclic2");
        }

        private void gpuHookAndJeeves_Click(object sender, EventArgs e)
        {
            StartProcessGPU("cudaHookAndJeeves2");
        }

        private void math_Click(object sender, EventArgs e)
        {
            form.ShowDialog();
        }

    }
}
