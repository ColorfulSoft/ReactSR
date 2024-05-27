//***************************************************************************************************
//* (C) ColorfulSoft corp., 2024. All rights reserved.
//* The code is available under the Apache-2.0 license. Read the License for details.
//***************************************************************************************************

using System;
using System.Drawing;
using System.Threading;
using System.Reflection;
using System.Windows.Forms;
using System.Drawing.Imaging;

namespace ColorfulSoft.ReactSR
{

    public sealed class MainForm : Form
    {

        private sealed class BeforeAfterPictureBox : UserControl
        {

            private Image __before;

            private Image __after;

            private float __slider;

            private bool __comparison;

            public Image Before
            {

                get
                {
                    return this.__before;
                }

                set
                {
                    this.__before = value;
                    this.Invalidate();
                }

            }

            public Image After
            {

                get
                {
                    return this.__after;
                }

                set
                {
                    this.__after = value;
                    this.Invalidate();
                }

            }

            public bool Comparison
            {

                get
                {
                    return this.__comparison;
                }

                set
                {
                    if(value != this.__comparison)
                    {
                        this.__comparison = value;
                        this.Invalidate();
                    }
                }

            }

            public BeforeAfterPictureBox()
            {
                this.Size = new Size(320, 240);
                this.__slider = 0.5f;
                this.Paint += this.ComparablePictureBox_Paint;
                this.MouseMove += this.ComparablePictureBox_MouseMove;
                this.MouseDown += this.ComparablePictureBox_MouseDown;
                this.Resize += this.ComparablePictureBox_Resize;
                this.DoubleBuffered = true;
            }

            private void ComparablePictureBox_Resize(object sender, EventArgs e)
            {
                this.Invalidate();
            }

            private void ComparablePictureBox_MouseDown(object sender, MouseEventArgs e)
            {
                if(!this.__comparison)
                {
                    return;
                }
                if(e.Button == MouseButtons.Left)
                {
                    this.Cursor = Cursors.VSplit;
                    this.__slider = e.X / (float)this.ClientSize.Width;
                    this.Invalidate();
                }
            }

            private void ComparablePictureBox_MouseMove(object sender, MouseEventArgs e)
            {
                if(!this.__comparison)
                {
                    return;
                }
                if(e.Button == MouseButtons.Left)
                {
                    this.Cursor = Cursors.VSplit;
                    this.__slider = e.X / (float)this.ClientSize.Width;
                    this.__slider = Math.Max(Math.Min(this.__slider, 1f), 0f);
                    this.Invalidate();
                    return;
                }
                if(Math.Abs(e.X - this.ClientSize.Width * this.__slider) <= 1f * this.CreateGraphics().DpiX / 96f)
                {
                    this.Cursor = Cursors.VSplit;
                }
                else
                {
                    this.Cursor = Cursors.Default;
                }
            }

            private void ComparablePictureBox_Paint(object sender, PaintEventArgs e)
            {
                Graphics g = e.Graphics;
                if(this.__comparison)
                {
                    if(this.__before != null)
                    {
                        float scale = Math.Max(this.__before.Width / (float)this.ClientSize.Width, this.__before.Height / (float)this.ClientSize.Height);
                        float src_width_scaled = this.__before.Width / scale;
                        float src_height_scaled = this.__before.Height / scale;
                        float left = (this.ClientSize.Width - src_width_scaled) / 2;
                        float top = (this.ClientSize.Height - src_height_scaled) / 2;
                        float dst_width = (this.__slider * this.ClientSize.Width - left) / (this.ClientSize.Width - left * 2);
                        dst_width = Math.Max(Math.Min(dst_width, 1f), 0f) * src_width_scaled;
                        float dst_height = src_height_scaled;
                        float src_width = (this.__slider * this.ClientSize.Width - left) / (this.ClientSize.Width - left * 2);
                        src_width = Math.Max(Math.Min(src_width, 1f), 0f) * this.__before.Width;
                        float src_height = this.__before.Height;
                        if((src_width >= 1) && (dst_width >= 1))
                        {
                            g.DrawImage(this.__before,
                                        new RectangleF(left, top, dst_width, dst_height),
                                        new RectangleF(0, 0, src_width, src_height),
                                        GraphicsUnit.Pixel);
                        }
                    }
                    if(this.__after != null)
                    {
                        float scale = Math.Max(this.__after.Width / (float)this.ClientSize.Width, this.__after.Height / (float)this.ClientSize.Height);
                        float src_width_scaled = this.__after.Width / scale;
                        float src_height_scaled = this.__after.Height / scale;
                        float right = (this.ClientSize.Width - src_width_scaled) / 2;
                        float top = (this.ClientSize.Height - src_height_scaled) / 2;
                        float dst_width = (this.ClientSize.Width - this.__slider * this.ClientSize.Width - right) / (this.ClientSize.Width - right * 2);
                        dst_width = Math.Max(Math.Min(dst_width, 1f), 0f) * src_width_scaled;
                        float dst_height = src_height_scaled;
                        float src_width = (this.ClientSize.Width - this.__slider * this.ClientSize.Width - right) / (this.ClientSize.Width - right * 2);
                        src_width = Math.Max(Math.Min(src_width, 1f), 0f) * this.__after.Width;
                        float src_height = this.__after.Height;
                        if((src_width >= 1) && (dst_width >= 1))
                        {
                            g.DrawImage(this.__after,
                                        new RectangleF(this.ClientSize.Width - dst_width - right, top, dst_width, dst_height),
                                        new RectangleF(this.__after.Width - src_width, 0, src_width, src_height),
                                        GraphicsUnit.Pixel);
                        }
                    }
                    using(Pen slider_pen = new Pen(Color.Red, 2 * g.DpiX / 96f))
                    {
                        g.DrawLine(slider_pen, this.ClientSize.Width * this.__slider, 0, this.ClientSize.Width * this.__slider, this.ClientSize.Height);
                    }
                }
                else
                {
                    if(this.__before != null)
                    {
                        float scale = Math.Max(this.__before.Width / (float)this.Width, this.__before.Height / (float)this.ClientSize.Height);
                        float top = (this.ClientSize.Height - this.__before.Height / scale) / 2;
                        float left = (this.ClientSize.Width - this.__before.Width / scale) / 2;
                        g.DrawImage(this.__before, new RectangleF(left, top, this.__before.Width / scale, this.__before.Height / scale), new RectangleF(0, 0, this.__before.Width, this.__before.Height), GraphicsUnit.Pixel);
                    }
                }
            }

        }

        private ToolStrip __tool_strip;

        private ToolStripMenuItem __open_image_tool_item;

        private ToolStripMenuItem __save_image_tool_item;

        private ToolStripMenuItem __start_process_tool_item;

        private ToolStripMenuItem __stop_process_tool_item;

        private ToolStripProgressBar __progress;

        private ToolStripMenuItem __compare_tool_item;

        private ToolStripMenuItem __about_tool_item;

        private MenuStrip __menu_strip;

        private ToolStripMenuItem __file_menu_item;

        private ToolStripMenuItem __open_file_menu_item;

        private ToolStripMenuItem __save_file_menu_item;

        private ToolStripMenuItem __cancel_file_menu_item;

        private ToolStripMenuItem __edit_menu_item;

        private ToolStripMenuItem __start_process_edit_menu_item;

        private ToolStripMenuItem __stop_process_edit_menu_item;

        private ToolStripMenuItem __compare_edit_menu_item;

        private ToolStripMenuItem __help_menu_item;

        private ToolStripMenuItem __about_help_menu_item;

        private BeforeAfterPictureBox __picture;

        private Bitmap __compare_icon;

        private Bitmap __not_compare_icon;

        private Bitmap __before;

        private Bitmap __after;

        private OpenFileDialog __open_file_dialog;

        private SaveFileDialog __save_file_dialog;

        private Thread __super_resolver_thread;

        private MainForm()
        {
            Assembly asm = Assembly.GetExecutingAssembly();
            Graphics g = this.CreateGraphics();
            float scale_x = g.DpiX / 96f;
            float scale_y = g.DpiY / 96f;
            this.__compare_icon = new Bitmap(new Bitmap(asm.GetManifestResourceStream("Compare.png")), (int)(16 * scale_x), (int)(16 * scale_y));
            this.__not_compare_icon = new Bitmap(new Bitmap(asm.GetManifestResourceStream("NotCompare.png")), (int)(16 * scale_x), (int)(16 * scale_y));
            this.__before = new Bitmap(asm.GetManifestResourceStream("Test.jpg"));
            this.__after = new Bitmap(asm.GetManifestResourceStream("TestSR.jpg"));
            // Main form
            this.Text = "ReactSR";
            this.Icon = Icon.FromHandle((new Bitmap(new Bitmap(asm.GetManifestResourceStream("ReactSR.png")), 256, 256)).GetHicon());
            this.MinimumSize = new Size((int)(640 * scale_x), (int)(480 * scale_y));
            this.FormClosed += this.MainForm_FormClosed;
            this.Resize += this.MainForm_Resize;
            // Tool strip
            this.__tool_strip = new ToolStrip();
            this.Controls.Add(this.__tool_strip);
            // Open image tool item
            this.__open_image_tool_item = new ToolStripMenuItem("", new Bitmap(new Bitmap(asm.GetManifestResourceStream("Open.png")), (int)(16 * scale_x), (int)(16 * scale_y)), this.__open_image);
            this.__open_image_tool_item.ImageScaling = ToolStripItemImageScaling.None;
            this.__tool_strip.Items.Add(this.__open_image_tool_item);
            // Save image tool item
            this.__save_image_tool_item = new ToolStripMenuItem("", new Bitmap(new Bitmap(asm.GetManifestResourceStream("Save.png")), (int)(16 * scale_x), (int)(16 * scale_y)), this.__save_image);
            this.__save_image_tool_item.ImageScaling = ToolStripItemImageScaling.None;
            this.__tool_strip.Items.Add(this.__save_image_tool_item);
            //----------
            this.__tool_strip.Items.Add(new ToolStripSeparator());
            // Start process tool item
            this.__start_process_tool_item = new ToolStripMenuItem("", new Bitmap(new Bitmap(asm.GetManifestResourceStream("Start.png")), (int)(16 * scale_x), (int)(16 * scale_y)), this.__start_process);
            this.__start_process_tool_item.ImageScaling = ToolStripItemImageScaling.None;
            this.__tool_strip.Items.Add(this.__start_process_tool_item);
            // Stop process tool item
            this.__stop_process_tool_item = new ToolStripMenuItem("", new Bitmap(new Bitmap(asm.GetManifestResourceStream("Stop.png")), (int)(16 * scale_x), (int)(16 * scale_y)), this.__stop_process);
            this.__stop_process_tool_item.ImageScaling = ToolStripItemImageScaling.None;
            this.__stop_process_tool_item.Enabled = false;
            this.__tool_strip.Items.Add(this.__stop_process_tool_item);
            // Progress
            this.__progress = new ToolStripProgressBar();
            this.__progress.AutoSize = false;
            this.__progress.Width = (int)(256 * scale_x);
            this.__progress.Height = this.__tool_strip.ClientSize.Height;
            this.__tool_strip.Items.Add(this.__progress);
            //----------
            this.__tool_strip.Items.Add(new ToolStripSeparator());
            // Compare tool item
            this.__compare_tool_item = new ToolStripMenuItem("", this.__not_compare_icon, this.__compare);
            this.__compare_tool_item.ImageScaling = ToolStripItemImageScaling.None;
            this.__tool_strip.Items.Add(this.__compare_tool_item);
            //----------
            this.__tool_strip.Items.Add(new ToolStripSeparator());
            // About tool item
            this.__about_tool_item = new ToolStripMenuItem("", new Bitmap(new Bitmap(asm.GetManifestResourceStream("Info.png")), (int)(16 * scale_x), (int)(16 * scale_y)), this.__show_help);
            this.__about_tool_item.ImageScaling = ToolStripItemImageScaling.None;
            this.__tool_strip.Items.Add(this.__about_tool_item);
            // Menu strip
            this.__menu_strip = new MenuStrip();
            this.Controls.Add(this.__menu_strip);
            // File menu item
            this.__file_menu_item = new ToolStripMenuItem("File");
            this.__menu_strip.Items.Add(this.__file_menu_item);
            // Open file menu item
            this.__open_file_menu_item = new ToolStripMenuItem("Open image", new Bitmap(new Bitmap(asm.GetManifestResourceStream("Open.png")), (int)(16 * scale_x), (int)(16 * scale_y)), this.__open_image);
            this.__open_file_menu_item.ImageScaling = ToolStripItemImageScaling.None;
            this.__open_file_menu_item.ShortcutKeys = Keys.Control | Keys.O;
            this.__file_menu_item.DropDownItems.Add(this.__open_file_menu_item);
            // Save file menu item
            this.__save_file_menu_item = new ToolStripMenuItem("Save image", new Bitmap(new Bitmap(asm.GetManifestResourceStream("Save.png")), (int)(16 * scale_x), (int)(16 * scale_y)), this.__save_image);
            this.__save_file_menu_item.ImageScaling = ToolStripItemImageScaling.None;
            this.__save_file_menu_item.ShortcutKeys = Keys.Control | Keys.S;
            this.__file_menu_item.DropDownItems.Add(this.__save_file_menu_item);
            //----------
            this.__file_menu_item.DropDownItems.Add(new ToolStripSeparator());
            // Cancel file menu item
            this.__cancel_file_menu_item = new ToolStripMenuItem("Exit", new Bitmap(new Bitmap(asm.GetManifestResourceStream("Cancel.png")), (int)(16 * scale_x), (int)(16 * scale_y)), this.__cancel);
            this.__cancel_file_menu_item.ImageScaling = ToolStripItemImageScaling.None;
            this.__file_menu_item.DropDownItems.Add(this.__cancel_file_menu_item);
            // Edit menu item
            this.__edit_menu_item = new ToolStripMenuItem("Edit");
            this.__menu_strip.Items.Add(this.__edit_menu_item);
            // Create super resolution edit menu item
            this.__start_process_edit_menu_item = new ToolStripMenuItem("Enhance", new Bitmap(new Bitmap(asm.GetManifestResourceStream("Start.png")), (int)(16 * scale_x), (int)(16 * scale_y)), this.__start_process);
            this.__start_process_edit_menu_item.ImageScaling = ToolStripItemImageScaling.None;
            this.__start_process_edit_menu_item.ShortcutKeys = Keys.F5;
            this.__edit_menu_item.DropDownItems.Add(this.__start_process_edit_menu_item);
            // Stop process edit menu item
            this.__stop_process_edit_menu_item = new ToolStripMenuItem("Stop", new Bitmap(new Bitmap(asm.GetManifestResourceStream("Stop.png")), (int)(16 * scale_x), (int)(16 * scale_y)), this.__stop_process);
            this.__stop_process_edit_menu_item.ImageScaling = ToolStripItemImageScaling.None;
            this.__stop_process_edit_menu_item.ShortcutKeys = Keys.Shift | Keys.F5;
            this.__stop_process_edit_menu_item.Enabled = false;
            this.__edit_menu_item.DropDownItems.Add(this.__stop_process_edit_menu_item);
            //----------
            this.__edit_menu_item.DropDownItems.Add(new ToolStripSeparator());
            // Compare edit menu item
            this.__compare_edit_menu_item = new ToolStripMenuItem("Compare", this.__not_compare_icon, this.__compare);
            this.__compare_edit_menu_item.ImageScaling = ToolStripItemImageScaling.None;
            this.__compare_edit_menu_item.ShortcutKeys = Keys.Control | Keys.Space;
            this.__edit_menu_item.DropDownItems.Add(this.__compare_edit_menu_item);
            // Help menu item
            this.__help_menu_item = new ToolStripMenuItem("Help");
            this.__menu_strip.Items.Add(this.__help_menu_item);
            // About help menu item
            this.__about_help_menu_item = new ToolStripMenuItem("About", new Bitmap(new Bitmap(asm.GetManifestResourceStream("Info.png")), (int)(16 * scale_x), (int)(16 * scale_y)), this.__show_help);
            this.__about_help_menu_item.ImageScaling = ToolStripItemImageScaling.None;
            this.__about_help_menu_item.ShortcutKeys = Keys.Control | Keys.I;
            this.__help_menu_item.DropDownItems.Add(this.__about_help_menu_item);
            // Scale menu strip
            this.__menu_strip.Scale(new SizeF(scale_x, scale_y));
            // Picture
            this.__picture = new BeforeAfterPictureBox();
            this.__picture.Width = this.ClientSize.Width;
            this.__picture.Height = this.ClientSize.Height - this.__tool_strip.Bottom;
            this.__picture.Top = this.__tool_strip.Bottom;
            this.__picture.Before = this.__before;
            this.__picture.After = this.__after;
            this.__picture.Comparison = true;
            this.__picture.BorderStyle = BorderStyle.FixedSingle;
            this.Controls.Add(this.__picture);
            // Open file dialog
            this.__open_file_dialog = new OpenFileDialog();
            this.__open_file_dialog.Filter = "Images|*.bmp; *.emf; *.exif; *.gif; *.ico; *.jpg; *.png; *.tiff; *.wmf|All files|*.*";
            // Save file dialog
            this.__save_file_dialog = new SaveFileDialog();
            this.__save_file_dialog.Filter = "BMP images|*.bmp|EMF images|*.emf|EXIF images|*.exif|GIF images|*.gif|ICO images|*.ico|JPG images|*.jpg|PNG images|*.png|TIFF images|*.tiff|WMF images|*.wmf";
        }

        private void MainForm_Resize(object sender, EventArgs e)
        {
            this.__picture.Width = this.ClientSize.Width;
            this.__picture.Height = this.ClientSize.Height - this.__tool_strip.Bottom;
            this.__picture.Top = this.__tool_strip.Bottom;
        }

        private void MainForm_FormClosed(object sender, FormClosedEventArgs e)
        {
            if(this.__super_resolver_thread != null)
            {
                if(this.__super_resolver_thread.ThreadState == ThreadState.Running)
                {
                    this.__super_resolver_thread.Abort();
                }
                this.__super_resolver_thread = null;
            }
        }

        private void __open_image(object sender, EventArgs e)
        {
        open:
            if(this.__open_file_dialog.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    this.__before = new Bitmap(this.__open_file_dialog.FileName);
                }
                catch(Exception ex)
                {
                    MessageBox.Show(ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    goto open;
                }
                this.__after = null;
                this.__open_file_dialog.FilterIndex = 0;
                this.__open_file_dialog.FileName = "";
                this.__save_file_menu_item.Enabled = false;
                this.__save_image_tool_item.Enabled = false;
                this.__picture.Before = this.__before;
                this.__picture.After = null;
                if(this.__picture.Comparison)
                {
                    this.__picture.Comparison = false;
                    this.__compare_edit_menu_item.Image = this.__compare_icon;
                    this.__compare_tool_item.Image = this.__compare_icon;
                }
                this.__compare_edit_menu_item.Enabled = false;
                this.__compare_tool_item.Enabled = false;
            }
        }

        private void __save_image(object sender, EventArgs e)
        {
        save:
            if(this.__save_file_dialog.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    switch(this.__save_file_dialog.FilterIndex)
                    {
                        case 1:
                        {
                            this.__after.Save(this.__save_file_dialog.FileName, ImageFormat.Bmp);
                            break;
                        }
                        case 2:
                        {
                            this.__after.Save(this.__save_file_dialog.FileName, ImageFormat.Emf);
                            break;
                        }
                        case 3:
                        {
                            this.__after.Save(this.__save_file_dialog.FileName, ImageFormat.Exif);
                            break;
                        }
                        case 4:
                        {
                            this.__after.Save(this.__save_file_dialog.FileName, ImageFormat.Gif);
                            break;
                        }
                        case 5:
                        {
                            this.__after.Save(this.__save_file_dialog.FileName, ImageFormat.Icon);
                            break;
                        }
                        case 6:
                        {
                            this.__after.Save(this.__save_file_dialog.FileName, ImageFormat.Jpeg);
                            break;
                        }
                        case 7:
                        {
                            this.__after.Save(this.__save_file_dialog.FileName, ImageFormat.Png);
                            break;
                        }
                        case 8:
                        {
                            this.__after.Save(this.__save_file_dialog.FileName, ImageFormat.Tiff);
                            break;
                        }
                        case 9:
                        {
                            this.__after.Save(this.__save_file_dialog.FileName, ImageFormat.Wmf);
                            break;
                        }
                    }
                }
                catch(Exception ex)
                {
                    MessageBox.Show(ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    goto save;
                }
                this.__save_file_dialog.FilterIndex = 0;
                this.__save_file_dialog.FileName = "";
            }
        }

        private void __cancel(object sender, EventArgs e)
        {
            this.Close();
        }

        private void __progress_setter(int progress)
        {
            this.Invoke((Action)(() =>
            {
                this.__progress.Value = progress;
            }));
        }

        private void __super_resolution_routine()
        {
            this.Invoke((Action)(() =>
            {
                this.__stop_process_edit_menu_item.Enabled = true;
                this.__stop_process_tool_item.Enabled = true;
            }));
            this.__after = SRViT.Instance.Process(this.__before.Clone(new Rectangle(new Point(0, 0), this.__before.Size), PixelFormat.Format24bppRgb), this.__progress_setter);
            this.Invoke((Action)(() =>
            {
                this.__save_file_menu_item.Enabled = true;
                this.__save_image_tool_item.Enabled = true;
                this.__start_process_edit_menu_item.Enabled = true;
                this.__start_process_tool_item.Enabled = true;
                this.__stop_process_edit_menu_item.Enabled = false;
                this.__stop_process_tool_item.Enabled = false;
                this.__picture.Before = this.__before;
                this.__picture.After = this.__after;
                this.__picture.Comparison = true;
                this.__compare_edit_menu_item.Image = this.__not_compare_icon;
                this.__compare_edit_menu_item.Enabled = true;
                this.__compare_tool_item.Image = this.__not_compare_icon;
                this.__compare_tool_item.Enabled = true;
            }));
            this.__super_resolver_thread = null;
        }

        private void __start_process(object sender, EventArgs e)
        {
            this.__after = null;
            this.__save_file_menu_item.Enabled = false;
            this.__save_image_tool_item.Enabled = false;
            this.__picture.After = null;
            if(this.__picture.Comparison)
            {
                this.__picture.Comparison = false;
                this.__compare_edit_menu_item.Image = this.__compare_icon;
                this.__compare_tool_item.Image = this.__compare_icon;
            }
            this.__compare_edit_menu_item.Enabled = false;
            this.__compare_tool_item.Enabled = false;
            this.__start_process_edit_menu_item.Enabled = false;
            this.__start_process_tool_item.Enabled = false;
            this.__super_resolver_thread = new Thread(this.__super_resolution_routine);
            this.__super_resolver_thread.Start();
        }

        private void __stop_process(object sender, EventArgs e)
        {
            if(this.__super_resolver_thread != null)
            {
                if(this.__super_resolver_thread.ThreadState == ThreadState.Running)
                {
                    this.__super_resolver_thread.Abort();
                }
                this.__super_resolver_thread = null;
            }
            this.__start_process_edit_menu_item.Enabled = true;
            this.__start_process_tool_item.Enabled = true;
            this.__stop_process_edit_menu_item.Enabled = false;
            this.__stop_process_tool_item.Enabled = false;
            this.__progress.Value = 0;
        }

        private void __compare(object sender, EventArgs e)
        {
            if(this.__picture.Comparison)
            {
                this.__picture.Comparison = false;
                this.__compare_tool_item.Image = this.__compare_icon;
                this.__compare_edit_menu_item.Image = this.__compare_icon;
                this.__picture.Before = this.__after;
            }
            else
            {
                this.__picture.Comparison = true;
                this.__compare_tool_item.Image = this.__not_compare_icon;
                this.__compare_edit_menu_item.Image = this.__not_compare_icon;
                this.__picture.Before = this.__before;
            }
        }

        private void __show_help(object sender, EventArgs e)
        {
            InfoForm.Instance.ShowDialog();
        }

        #region Singleton

        /// <summary>
        /// An instance of the main form.
        /// </summary>
        public static readonly MainForm Instance;

        /// <summary>
        /// Provides singleton initialization.
        /// </summary>
        static MainForm()
        {
            Instance = new MainForm();
        }

        #endregion

    }

}