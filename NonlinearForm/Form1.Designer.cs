namespace NonlinearForm
{
    partial class Form1
    {
        /// <summary>
        /// Требуется переменная конструктора.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Освободить все используемые ресурсы.
        /// </summary>
        /// <param name="disposing">истинно, если управляемый ресурс должен быть удален; иначе ложно.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Код, автоматически созданный конструктором форм Windows

        /// <summary>
        /// Обязательный метод для поддержки конструктора - не изменяйте
        /// содержимое данного метода при помощи редактора кода.
        /// </summary>
        private void InitializeComponent()
        {
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.запуститьToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.математическоеУравнениеToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.gPUToolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
            this.методРешетокToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.циклическийПокоординатныйСпускToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.алгритмХукаИДживсаToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.cPUToolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
            this.методРешетокToolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
            this.циклическийПокоординатныйСпускToolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
            this.алгритмХукаИДживсаToolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
            this.очиститьЛогToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem1 = new System.Windows.Forms.ToolStripSeparator();
            this.выходToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.оптимизацияToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.cudaToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.экспериментToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.построитьГрафикиToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.справкаToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.оПрограммеToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.справкаToolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripContainer1 = new System.Windows.Forms.ToolStripContainer();
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.numericUpDown2 = new System.Windows.Forms.NumericUpDown();
            this.numericUpDown1 = new System.Windows.Forms.NumericUpDown();
            this.label2 = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.richTextBox1 = new System.Windows.Forms.RichTextBox();
            this.menuStrip1.SuspendLayout();
            this.toolStripContainer1.ContentPanel.SuspendLayout();
            this.toolStripContainer1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            this.groupBox1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown1)).BeginInit();
            this.SuspendLayout();
            // 
            // menuStrip1
            // 
            this.menuStrip1.ImageScalingSize = new System.Drawing.Size(24, 24);
            this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.запуститьToolStripMenuItem,
            this.оптимизацияToolStripMenuItem,
            this.справкаToolStripMenuItem});
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Padding = new System.Windows.Forms.Padding(9, 3, 0, 3);
            this.menuStrip1.Size = new System.Drawing.Size(1088, 35);
            this.menuStrip1.TabIndex = 0;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // запуститьToolStripMenuItem
            // 
            this.запуститьToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.математическоеУравнениеToolStripMenuItem,
            this.gPUToolStripMenuItem1,
            this.cPUToolStripMenuItem1,
            this.очиститьЛогToolStripMenuItem,
            this.toolStripMenuItem1,
            this.выходToolStripMenuItem});
            this.запуститьToolStripMenuItem.Name = "запуститьToolStripMenuItem";
            this.запуститьToolStripMenuItem.Size = new System.Drawing.Size(65, 29);
            this.запуститьToolStripMenuItem.Text = "Файл";
            // 
            // математическоеУравнениеToolStripMenuItem
            // 
            this.математическоеУравнениеToolStripMenuItem.Name = "математическоеУравнениеToolStripMenuItem";
            this.математическоеУравнениеToolStripMenuItem.Size = new System.Drawing.Size(325, 30);
            this.математическоеУравнениеToolStripMenuItem.Text = "Математическое уравнение";
            this.математическоеУравнениеToolStripMenuItem.Click += new System.EventHandler(this.math_Click);
            // 
            // gPUToolStripMenuItem1
            // 
            this.gPUToolStripMenuItem1.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.методРешетокToolStripMenuItem,
            this.циклическийПокоординатныйСпускToolStripMenuItem,
            this.алгритмХукаИДживсаToolStripMenuItem});
            this.gPUToolStripMenuItem1.Name = "gPUToolStripMenuItem1";
            this.gPUToolStripMenuItem1.Size = new System.Drawing.Size(325, 30);
            this.gPUToolStripMenuItem1.Text = "GPU";
            // 
            // методРешетокToolStripMenuItem
            // 
            this.методРешетокToolStripMenuItem.Name = "методРешетокToolStripMenuItem";
            this.методРешетокToolStripMenuItem.Size = new System.Drawing.Size(400, 30);
            this.методРешетокToolStripMenuItem.Text = "Метод решеток";
            this.методРешетокToolStripMenuItem.Click += new System.EventHandler(this.gpudichotomy_Click);
            // 
            // циклическийПокоординатныйСпускToolStripMenuItem
            // 
            this.циклическийПокоординатныйСпускToolStripMenuItem.Name = "циклическийПокоординатныйСпускToolStripMenuItem";
            this.циклическийПокоординатныйСпускToolStripMenuItem.Size = new System.Drawing.Size(400, 30);
            this.циклическийПокоординатныйСпускToolStripMenuItem.Text = "Циклический покоординатный спуск";
            this.циклическийПокоординатныйСпускToolStripMenuItem.Click += new System.EventHandler(this.gpucyclic_Click);
            // 
            // алгритмХукаИДживсаToolStripMenuItem
            // 
            this.алгритмХукаИДживсаToolStripMenuItem.Name = "алгритмХукаИДживсаToolStripMenuItem";
            this.алгритмХукаИДживсаToolStripMenuItem.Size = new System.Drawing.Size(400, 30);
            this.алгритмХукаИДживсаToolStripMenuItem.Text = "Алгритм Хука и Дживса";
            this.алгритмХукаИДживсаToolStripMenuItem.Click += new System.EventHandler(this.gpuHookAndJeeves_Click);
            // 
            // cPUToolStripMenuItem1
            // 
            this.cPUToolStripMenuItem1.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.методРешетокToolStripMenuItem1,
            this.циклическийПокоординатныйСпускToolStripMenuItem1,
            this.алгритмХукаИДживсаToolStripMenuItem1});
            this.cPUToolStripMenuItem1.Name = "cPUToolStripMenuItem1";
            this.cPUToolStripMenuItem1.Size = new System.Drawing.Size(325, 30);
            this.cPUToolStripMenuItem1.Text = "CPU";
            // 
            // методРешетокToolStripMenuItem1
            // 
            this.методРешетокToolStripMenuItem1.Name = "методРешетокToolStripMenuItem1";
            this.методРешетокToolStripMenuItem1.Size = new System.Drawing.Size(400, 30);
            this.методРешетокToolStripMenuItem1.Text = "Метод решеток";
            this.методРешетокToolStripMenuItem1.Click += new System.EventHandler(this.cpudichotomy_Click);
            // 
            // циклическийПокоординатныйСпускToolStripMenuItem1
            // 
            this.циклическийПокоординатныйСпускToolStripMenuItem1.Name = "циклическийПокоординатныйСпускToolStripMenuItem1";
            this.циклическийПокоординатныйСпускToolStripMenuItem1.Size = new System.Drawing.Size(400, 30);
            this.циклическийПокоординатныйСпускToolStripMenuItem1.Text = "Циклический покоординатный спуск";
            this.циклическийПокоординатныйСпускToolStripMenuItem1.Click += new System.EventHandler(this.cpucyclic_Click);
            // 
            // алгритмХукаИДживсаToolStripMenuItem1
            // 
            this.алгритмХукаИДживсаToolStripMenuItem1.Name = "алгритмХукаИДживсаToolStripMenuItem1";
            this.алгритмХукаИДживсаToolStripMenuItem1.Size = new System.Drawing.Size(400, 30);
            this.алгритмХукаИДживсаToolStripMenuItem1.Text = "Алгритм Хука и Дживса";
            this.алгритмХукаИДживсаToolStripMenuItem1.Click += new System.EventHandler(this.cpuHookAndJeeves_Click);
            // 
            // очиститьЛогToolStripMenuItem
            // 
            this.очиститьЛогToolStripMenuItem.Name = "очиститьЛогToolStripMenuItem";
            this.очиститьЛогToolStripMenuItem.Size = new System.Drawing.Size(325, 30);
            this.очиститьЛогToolStripMenuItem.Text = "Очистить лог";
            this.очиститьЛогToolStripMenuItem.Click += new System.EventHandler(this.clean_Click);
            // 
            // toolStripMenuItem1
            // 
            this.toolStripMenuItem1.Name = "toolStripMenuItem1";
            this.toolStripMenuItem1.Size = new System.Drawing.Size(322, 6);
            // 
            // выходToolStripMenuItem
            // 
            this.выходToolStripMenuItem.Name = "выходToolStripMenuItem";
            this.выходToolStripMenuItem.Size = new System.Drawing.Size(325, 30);
            this.выходToolStripMenuItem.Text = "Выход";
            this.выходToolStripMenuItem.Click += new System.EventHandler(this.выходToolStripMenuItem_Click);
            // 
            // оптимизацияToolStripMenuItem
            // 
            this.оптимизацияToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.cudaToolStripMenuItem,
            this.экспериментToolStripMenuItem});
            this.оптимизацияToolStripMenuItem.Name = "оптимизацияToolStripMenuItem";
            this.оптимизацияToolStripMenuItem.Size = new System.Drawing.Size(83, 29);
            this.оптимизацияToolStripMenuItem.Text = "Сервис";
            // 
            // cudaToolStripMenuItem
            // 
            this.cudaToolStripMenuItem.Name = "cudaToolStripMenuItem";
            this.cudaToolStripMenuItem.Size = new System.Drawing.Size(247, 30);
            this.cudaToolStripMenuItem.Text = "Оптимизация GPU";
            // 
            // экспериментToolStripMenuItem
            // 
            this.экспериментToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.построитьГрафикиToolStripMenuItem});
            this.экспериментToolStripMenuItem.Name = "экспериментToolStripMenuItem";
            this.экспериментToolStripMenuItem.Size = new System.Drawing.Size(247, 30);
            this.экспериментToolStripMenuItem.Text = "Эксперимент";
            // 
            // построитьГрафикиToolStripMenuItem
            // 
            this.построитьГрафикиToolStripMenuItem.Enabled = false;
            this.построитьГрафикиToolStripMenuItem.Name = "построитьГрафикиToolStripMenuItem";
            this.построитьГрафикиToolStripMenuItem.Size = new System.Drawing.Size(254, 30);
            this.построитьГрафикиToolStripMenuItem.Text = "построить графики";
            // 
            // справкаToolStripMenuItem
            // 
            this.справкаToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.оПрограммеToolStripMenuItem,
            this.справкаToolStripMenuItem1});
            this.справкаToolStripMenuItem.Name = "справкаToolStripMenuItem";
            this.справкаToolStripMenuItem.Size = new System.Drawing.Size(93, 29);
            this.справкаToolStripMenuItem.Text = "Справка";
            // 
            // оПрограммеToolStripMenuItem
            // 
            this.оПрограммеToolStripMenuItem.Name = "оПрограммеToolStripMenuItem";
            this.оПрограммеToolStripMenuItem.Size = new System.Drawing.Size(210, 30);
            this.оПрограммеToolStripMenuItem.Text = "О программе";
            // 
            // справкаToolStripMenuItem1
            // 
            this.справкаToolStripMenuItem1.Name = "справкаToolStripMenuItem1";
            this.справкаToolStripMenuItem1.Size = new System.Drawing.Size(210, 30);
            this.справкаToolStripMenuItem1.Text = "Справка";
            // 
            // toolStripContainer1
            // 
            // 
            // toolStripContainer1.ContentPanel
            // 
            this.toolStripContainer1.ContentPanel.Controls.Add(this.splitContainer1);
            this.toolStripContainer1.ContentPanel.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.toolStripContainer1.ContentPanel.Size = new System.Drawing.Size(1088, 538);
            this.toolStripContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.toolStripContainer1.Location = new System.Drawing.Point(0, 35);
            this.toolStripContainer1.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.toolStripContainer1.Name = "toolStripContainer1";
            this.toolStripContainer1.Size = new System.Drawing.Size(1088, 563);
            this.toolStripContainer1.TabIndex = 1;
            this.toolStripContainer1.Text = "toolStripContainer1";
            // 
            // splitContainer1
            // 
            this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer1.Location = new System.Drawing.Point(0, 0);
            this.splitContainer1.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.splitContainer1.Name = "splitContainer1";
            // 
            // splitContainer1.Panel1
            // 
            this.splitContainer1.Panel1.Controls.Add(this.groupBox1);
            // 
            // splitContainer1.Panel2
            // 
            this.splitContainer1.Panel2.Controls.Add(this.richTextBox1);
            this.splitContainer1.Size = new System.Drawing.Size(1088, 538);
            this.splitContainer1.SplitterDistance = 361;
            this.splitContainer1.SplitterWidth = 6;
            this.splitContainer1.TabIndex = 0;
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.numericUpDown2);
            this.groupBox1.Controls.Add(this.numericUpDown1);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Controls.Add(this.label1);
            this.groupBox1.Dock = System.Windows.Forms.DockStyle.Top;
            this.groupBox1.Location = new System.Drawing.Point(0, 0);
            this.groupBox1.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Padding = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.groupBox1.Size = new System.Drawing.Size(361, 328);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Параметры запуска";
            // 
            // numericUpDown2
            // 
            this.numericUpDown2.Location = new System.Drawing.Point(100, 82);
            this.numericUpDown2.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.numericUpDown2.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numericUpDown2.Name = "numericUpDown2";
            this.numericUpDown2.Size = new System.Drawing.Size(100, 26);
            this.numericUpDown2.TabIndex = 2;
            this.numericUpDown2.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.numericUpDown2.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            // 
            // numericUpDown1
            // 
            this.numericUpDown1.Location = new System.Drawing.Point(100, 42);
            this.numericUpDown1.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.numericUpDown1.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numericUpDown1.Name = "numericUpDown1";
            this.numericUpDown1.Size = new System.Drawing.Size(100, 26);
            this.numericUpDown1.TabIndex = 2;
            this.numericUpDown1.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.numericUpDown1.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(9, 85);
            this.label2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(49, 20);
            this.label2.TabIndex = 1;
            this.label2.Text = "нити:";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(9, 45);
            this.label1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(62, 20);
            this.label1.TabIndex = 0;
            this.label1.Text = "блоки: ";
            // 
            // richTextBox1
            // 
            this.richTextBox1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.richTextBox1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.richTextBox1.Location = new System.Drawing.Point(0, 0);
            this.richTextBox1.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.richTextBox1.Name = "richTextBox1";
            this.richTextBox1.ReadOnly = true;
            this.richTextBox1.Size = new System.Drawing.Size(721, 538);
            this.richTextBox1.TabIndex = 0;
            this.richTextBox1.Text = "";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1088, 598);
            this.Controls.Add(this.toolStripContainer1);
            this.Controls.Add(this.menuStrip1);
            this.MainMenuStrip = this.menuStrip1;
            this.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.Name = "Form1";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Решение задач нелинейного программирования";
            this.menuStrip1.ResumeLayout(false);
            this.menuStrip1.PerformLayout();
            this.toolStripContainer1.ContentPanel.ResumeLayout(false);
            this.toolStripContainer1.ResumeLayout(false);
            this.toolStripContainer1.PerformLayout();
            this.splitContainer1.Panel1.ResumeLayout(false);
            this.splitContainer1.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
            this.splitContainer1.ResumeLayout(false);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown1)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.ToolStripMenuItem запуститьToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem gPUToolStripMenuItem1;
        private System.Windows.Forms.ToolStripMenuItem методРешетокToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem циклическийПокоординатныйСпускToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem алгритмХукаИДживсаToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem cPUToolStripMenuItem1;
        private System.Windows.Forms.ToolStripMenuItem методРешетокToolStripMenuItem1;
        private System.Windows.Forms.ToolStripMenuItem циклическийПокоординатныйСпускToolStripMenuItem1;
        private System.Windows.Forms.ToolStripMenuItem алгритмХукаИДживсаToolStripMenuItem1;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem1;
        private System.Windows.Forms.ToolStripMenuItem выходToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem оптимизацияToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem cudaToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem справкаToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem оПрограммеToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem справкаToolStripMenuItem1;
        private System.Windows.Forms.ToolStripContainer toolStripContainer1;
        private System.Windows.Forms.SplitContainer splitContainer1;
        private System.Windows.Forms.RichTextBox richTextBox1;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.ToolStripMenuItem экспериментToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem построитьГрафикиToolStripMenuItem;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.NumericUpDown numericUpDown2;
        private System.Windows.Forms.NumericUpDown numericUpDown1;
        private System.Windows.Forms.ToolStripMenuItem очиститьЛогToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem математическоеУравнениеToolStripMenuItem;
    }
}

