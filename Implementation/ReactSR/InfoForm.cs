//***************************************************************************************************
//* (C) ColorfulSoft corp., 2024. All rights reserved.
//* The code is available under the Apache-2.0 license. Read the License for details.
//***************************************************************************************************

using System.Drawing;
using System.Reflection;
using System.Windows.Forms;

namespace ColorfulSoft.ReactSR
{

    public sealed class InfoForm : Form
    {

        private PictureBox __logo;

        private Label __text;

        private InfoForm()
        {
            Assembly asm = Assembly.GetExecutingAssembly();
            Graphics g = this.CreateGraphics();
            float scale_x = g.DpiX / 96f;
            float scale_y = g.DpiY / 96f;
            // Form
            this.Text = "About ReactSR";
            this.ShowIcon = false;
            this.ShowInTaskbar = false;
            this.FormBorderStyle = FormBorderStyle.FixedSingle;
            this.ClientSize = new Size((int)(320 * scale_x), (int)((256 + Label.DefaultFont.Height) * scale_y));
            this.MaximumSize = this.Size;
            this.MinimumSize = this.Size;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.TopMost = true;
            // Logo
            this.__logo = new PictureBox();
            this.__logo.Image = new Bitmap(asm.GetManifestResourceStream("ReactSR.png"));
            this.__logo.SizeMode = PictureBoxSizeMode.Zoom;
            this.__logo.Left = (int)(32 * scale_x);
            this.__logo.Size = new Size((int)(256 * scale_x), (int)(256 * scale_y));
            this.Controls.Add(this.__logo);
            // Text
            this.__text = new Label();
            this.__text.Text = "Copyright 2024 Gleb S. Brykin for ColorfulSoft corp.";
            this.__text.Size = g.MeasureString(this.__text.Text, this.__text.Font).ToSize();
            this.__text.Top = (int)(256 * scale_y);
            this.__text.Left = (int)((this.ClientSize.Width - this.__text.Width) / 2);
            this.Controls.Add(this.__text);
        }

        #region Singleton

        public static readonly InfoForm Instance;

        static InfoForm()
        {
            Instance = new InfoForm();
        }

        #endregion

    }

}