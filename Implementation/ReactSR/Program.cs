//***************************************************************************************************
//* (C) ColorfulSoft corp., 2024. All rights reserved.
//* The code is available under the Apache-2.0 license. Read the License for details.
//***************************************************************************************************

using System;
using System.Windows.Forms;

namespace ColorfulSoft.ReactSR
{

    public static class Program
    {

        [STAThread]
        public static void Main(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(MainForm.Instance);
        }

    }

}