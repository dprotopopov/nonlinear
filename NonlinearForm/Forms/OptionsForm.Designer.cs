namespace NonlinearForm.Forms
{
    partial class OptionsForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this._matrixControlM = new NonlinearForm.Forms.MatrixControl();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this._matrixControlB = new NonlinearForm.Forms.MatrixControl();
            this._matrixControlA = new NonlinearForm.Forms.MatrixControl();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.button2 = new System.Windows.Forms.Button();
            this.button1 = new System.Windows.Forms.Button();
            this._matrixControlF = new NonlinearForm.Forms.MatrixControl();
            this.numericUpDown1 = new System.Windows.Forms.NumericUpDown();
            this.groupBox4 = new System.Windows.Forms.GroupBox();
            this.button3 = new System.Windows.Forms.Button();
            this.button4 = new System.Windows.Forms.Button();
            this._matrixControlW = new NonlinearForm.Forms.MatrixControl();
            this.button5 = new System.Windows.Forms.Button();
            this.button6 = new System.Windows.Forms.Button();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.groupBox3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown1)).BeginInit();
            this.groupBox4.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this._matrixControlM);
            this.groupBox1.Location = new System.Drawing.Point(61, 66);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(741, 124);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Число сегментов";
            // 
            // _matrixControlM
            // 
            this._matrixControlM.Columns = 2;
            this._matrixControlM.Location = new System.Drawing.Point(57, 25);
            this._matrixControlM.Name = "_matrixControlM";
            this._matrixControlM.ReadOnly = false;
            this._matrixControlM.Rows = 1;
            this._matrixControlM.Size = new System.Drawing.Size(537, 81);
            this._matrixControlM.TabIndex = 0;
            new NonlinearForm.Vector<decimal>().Add(new decimal(new int[] {
                0,
                0,
                0,
                0}));
            new NonlinearForm.Vector<decimal>().Add(new decimal(new int[] {
                0,
                0,
                0,
                0}));
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this._matrixControlB);
            this.groupBox2.Controls.Add(this._matrixControlA);
            this.groupBox2.Location = new System.Drawing.Point(61, 196);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(741, 229);
            this.groupBox2.TabIndex = 1;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Минимальные/Максимальные  координаты";
            // 
            // _matrixControlB
            // 
            this._matrixControlB.Columns = 2;
            this._matrixControlB.Location = new System.Drawing.Point(57, 133);
            this._matrixControlB.Name = "_matrixControlB";
            this._matrixControlB.ReadOnly = false;
            this._matrixControlB.Rows = 1;
            this._matrixControlB.Size = new System.Drawing.Size(537, 77);
            this._matrixControlB.TabIndex = 1;
            new NonlinearForm.Vector<decimal>().Add(new decimal(new int[] {
                0,
                0,
                0,
                0}));
            new NonlinearForm.Vector<decimal>().Add(new decimal(new int[] {
                0,
                0,
                0,
                0}));
            // 
            // _matrixControlA
            // 
            this._matrixControlA.Columns = 2;
            this._matrixControlA.Location = new System.Drawing.Point(57, 34);
            this._matrixControlA.Name = "_matrixControlA";
            this._matrixControlA.ReadOnly = false;
            this._matrixControlA.Rows = 1;
            this._matrixControlA.Size = new System.Drawing.Size(537, 82);
            this._matrixControlA.TabIndex = 0;
            new NonlinearForm.Vector<decimal>().Add(new decimal(new int[] {
                0,
                0,
                0,
                0}));
            new NonlinearForm.Vector<decimal>().Add(new decimal(new int[] {
                0,
                0,
                0,
                0}));
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.button2);
            this.groupBox3.Controls.Add(this.button1);
            this.groupBox3.Controls.Add(this._matrixControlF);
            this.groupBox3.Location = new System.Drawing.Point(61, 431);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Size = new System.Drawing.Size(741, 207);
            this.groupBox3.TabIndex = 2;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Ограничения";
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(640, 54);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(75, 23);
            this.button2.TabIndex = 2;
            this.button2.Text = "Remove";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(640, 25);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(75, 23);
            this.button1.TabIndex = 1;
            this.button1.Text = "Add";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // _matrixControlF
            // 
            this._matrixControlF.Columns = 3;
            this._matrixControlF.Location = new System.Drawing.Point(57, 25);
            this._matrixControlF.Name = "_matrixControlF";
            this._matrixControlF.ReadOnly = false;
            this._matrixControlF.Rows = 1;
            this._matrixControlF.Size = new System.Drawing.Size(537, 156);
            this._matrixControlF.TabIndex = 0;
            new NonlinearForm.Vector<decimal>().Add(new decimal(new int[] {
                0,
                0,
                0,
                0}));
            new NonlinearForm.Vector<decimal>().Add(new decimal(new int[] {
                0,
                0,
                0,
                0}));
            new NonlinearForm.Vector<decimal>().Add(new decimal(new int[] {
                0,
                0,
                0,
                0}));
            // 
            // numericUpDown1
            // 
            this.numericUpDown1.Location = new System.Drawing.Point(288, 23);
            this.numericUpDown1.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numericUpDown1.Name = "numericUpDown1";
            this.numericUpDown1.Size = new System.Drawing.Size(120, 26);
            this.numericUpDown1.TabIndex = 0;
            this.numericUpDown1.Value = new decimal(new int[] {
            2,
            0,
            0,
            0});
            this.numericUpDown1.ValueChanged += new System.EventHandler(this.numericUpDown1_ValueChanged);
            // 
            // groupBox4
            // 
            this.groupBox4.Controls.Add(this.button3);
            this.groupBox4.Controls.Add(this.button4);
            this.groupBox4.Controls.Add(this._matrixControlW);
            this.groupBox4.Location = new System.Drawing.Point(61, 655);
            this.groupBox4.Name = "groupBox4";
            this.groupBox4.Size = new System.Drawing.Size(741, 203);
            this.groupBox4.TabIndex = 3;
            this.groupBox4.TabStop = false;
            this.groupBox4.Text = "Искомая функция";
            // 
            // button3
            // 
            this.button3.Location = new System.Drawing.Point(640, 54);
            this.button3.Name = "button3";
            this.button3.Size = new System.Drawing.Size(75, 23);
            this.button3.TabIndex = 4;
            this.button3.Text = "Remove";
            this.button3.UseVisualStyleBackColor = true;
            this.button3.Click += new System.EventHandler(this.button3_Click);
            // 
            // button4
            // 
            this.button4.Location = new System.Drawing.Point(640, 25);
            this.button4.Name = "button4";
            this.button4.Size = new System.Drawing.Size(75, 23);
            this.button4.TabIndex = 3;
            this.button4.Text = "Add";
            this.button4.UseVisualStyleBackColor = true;
            this.button4.Click += new System.EventHandler(this.button4_Click);
            // 
            // _matrixControlW
            // 
            this._matrixControlW.Columns = 3;
            this._matrixControlW.Location = new System.Drawing.Point(57, 25);
            this._matrixControlW.Name = "_matrixControlW";
            this._matrixControlW.ReadOnly = false;
            this._matrixControlW.Rows = 1;
            this._matrixControlW.Size = new System.Drawing.Size(537, 164);
            this._matrixControlW.TabIndex = 0;
            new NonlinearForm.Vector<decimal>().Add(new decimal(new int[] {
                0,
                0,
                0,
                0}));
            new NonlinearForm.Vector<decimal>().Add(new decimal(new int[] {
                0,
                0,
                0,
                0}));
            new NonlinearForm.Vector<decimal>().Add(new decimal(new int[] {
                0,
                0,
                0,
                0}));
            // 
            // button5
            // 
            this.button5.Location = new System.Drawing.Point(580, 864);
            this.button5.Name = "button5";
            this.button5.Size = new System.Drawing.Size(75, 23);
            this.button5.TabIndex = 4;
            this.button5.Text = "Open";
            this.button5.UseVisualStyleBackColor = true;
            this.button5.Click += new System.EventHandler(this.button5_Click);
            // 
            // button6
            // 
            this.button6.Location = new System.Drawing.Point(701, 864);
            this.button6.Name = "button6";
            this.button6.Size = new System.Drawing.Size(75, 23);
            this.button6.TabIndex = 5;
            this.button6.Text = "Save";
            this.button6.UseVisualStyleBackColor = true;
            this.button6.Click += new System.EventHandler(this.button6_Click);
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "math.txt";
            // 
            // saveFileDialog1
            // 
            this.saveFileDialog1.FileName = "math.txt";
            // 
            // OptionsForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(925, 909);
            this.Controls.Add(this.button6);
            this.Controls.Add(this.button5);
            this.Controls.Add(this.groupBox4);
            this.Controls.Add(this.numericUpDown1);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.Name = "OptionsForm";
            this.Text = "Form2";
            this.groupBox1.ResumeLayout(false);
            this.groupBox2.ResumeLayout(false);
            this.groupBox3.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown1)).EndInit();
            this.groupBox4.ResumeLayout(false);
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.GroupBox groupBox3;
        private System.Windows.Forms.NumericUpDown numericUpDown1;
        private MatrixControl _matrixControlA;
        private MatrixControl _matrixControlM;
        private MatrixControl _matrixControlF;
        private System.Windows.Forms.GroupBox groupBox4;
        private MatrixControl _matrixControlW;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.Button button3;
        private System.Windows.Forms.Button button4;
        private System.Windows.Forms.Button button5;
        private System.Windows.Forms.Button button6;
        private System.Windows.Forms.OpenFileDialog openFileDialog1;
        private System.Windows.Forms.SaveFileDialog saveFileDialog1;
        private MatrixControl _matrixControlB;
    }
}