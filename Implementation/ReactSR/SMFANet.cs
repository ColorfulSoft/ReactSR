//***************************************************************************************************
//* (C) ColorfulSoft corp., 2024. All rights reserved.
//* The code is available under the Apache-2.0 license. Read the License for details.
//***************************************************************************************************

using System;
using System.IO;
using System.Drawing;
using System.Reflection;
using System.Threading.Tasks;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace ColorfulSoft.ReactSR
{

    internal unsafe sealed class SMFANet
    {

        public sealed class Tensor
        {

            public float* data_ptr;

            public int* shape_ptr;

            public int ndim;

            public int numel;

            public Tensor(params int[] shape)
            {
                this.shape_ptr = (int*)Marshal.AllocHGlobal(shape.Length * sizeof(int));
                this.ndim = shape.Length;
                this.numel = 1;
                for(int i = 0; i < shape.Length; ++i)
                {
                    this.shape_ptr[i] = shape[i];
                    this.numel *= shape[i];
                }
                try
                {
                    this.data_ptr = (float*)Marshal.AllocCoTaskMem(this.numel * sizeof(float));
                }
                catch
                {
                    Marshal.FreeHGlobal((IntPtr)this.shape_ptr);
                    throw;
                }
            }

            ~Tensor()
            {
                if(this.shape_ptr != null)
                {
                    Marshal.FreeHGlobal((IntPtr)this.shape_ptr);
                    this.shape_ptr = null;
                }
                if(this.data_ptr != null)
                {
                    Marshal.FreeCoTaskMem((IntPtr)this.data_ptr);
                    this.data_ptr = null;
                }
            }

        }

        private readonly Tensor __to_feat_weight;

        private readonly Tensor __to_feat_bias;

        private readonly Tensor __feats_0_smfa_alpha;

        private readonly Tensor __feats_0_smfa_belt;

        private readonly Tensor __feats_0_smfa_linear_0_weight;

        private readonly Tensor __feats_0_smfa_linear_0_bias;

        private readonly Tensor __feats_0_smfa_linear_1_weight;

        private readonly Tensor __feats_0_smfa_linear_1_bias;

        private readonly Tensor __feats_0_smfa_linear_2_weight;

        private readonly Tensor __feats_0_smfa_linear_2_bias;

        private readonly Tensor __feats_0_smfa_lde_conv_0_0_weight;

        private readonly Tensor __feats_0_smfa_lde_conv_0_0_bias;

        private readonly Tensor __feats_0_smfa_lde_conv_0_1_weight;

        private readonly Tensor __feats_0_smfa_lde_conv_0_1_bias;

        private readonly Tensor __feats_0_smfa_lde_conv_1_weight;

        private readonly Tensor __feats_0_smfa_lde_conv_1_bias;

        private readonly Tensor __feats_0_smfa_dw_conv_weight;

        private readonly Tensor __feats_0_smfa_dw_conv_bias;

        private readonly Tensor __feats_0_pcfn_conv_0_weight;

        private readonly Tensor __feats_0_pcfn_conv_0_bias;

        private readonly Tensor __feats_0_pcfn_conv_1_weight;

        private readonly Tensor __feats_0_pcfn_conv_1_bias;

        private readonly Tensor __feats_0_pcfn_conv_2_weight;

        private readonly Tensor __feats_0_pcfn_conv_2_bias;

        private readonly Tensor __feats_1_smfa_alpha;

        private readonly Tensor __feats_1_smfa_belt;

        private readonly Tensor __feats_1_smfa_linear_0_weight;

        private readonly Tensor __feats_1_smfa_linear_0_bias;

        private readonly Tensor __feats_1_smfa_linear_1_weight;

        private readonly Tensor __feats_1_smfa_linear_1_bias;

        private readonly Tensor __feats_1_smfa_linear_2_weight;

        private readonly Tensor __feats_1_smfa_linear_2_bias;

        private readonly Tensor __feats_1_smfa_lde_conv_0_0_weight;

        private readonly Tensor __feats_1_smfa_lde_conv_0_0_bias;

        private readonly Tensor __feats_1_smfa_lde_conv_0_1_weight;

        private readonly Tensor __feats_1_smfa_lde_conv_0_1_bias;

        private readonly Tensor __feats_1_smfa_lde_conv_1_weight;

        private readonly Tensor __feats_1_smfa_lde_conv_1_bias;

        private readonly Tensor __feats_1_smfa_dw_conv_weight;

        private readonly Tensor __feats_1_smfa_dw_conv_bias;

        private readonly Tensor __feats_1_pcfn_conv_0_weight;

        private readonly Tensor __feats_1_pcfn_conv_0_bias;

        private readonly Tensor __feats_1_pcfn_conv_1_weight;

        private readonly Tensor __feats_1_pcfn_conv_1_bias;

        private readonly Tensor __feats_1_pcfn_conv_2_weight;

        private readonly Tensor __feats_1_pcfn_conv_2_bias;

        private readonly Tensor __feats_2_smfa_alpha;

        private readonly Tensor __feats_2_smfa_belt;

        private readonly Tensor __feats_2_smfa_linear_0_weight;

        private readonly Tensor __feats_2_smfa_linear_0_bias;

        private readonly Tensor __feats_2_smfa_linear_1_weight;

        private readonly Tensor __feats_2_smfa_linear_1_bias;

        private readonly Tensor __feats_2_smfa_linear_2_weight;

        private readonly Tensor __feats_2_smfa_linear_2_bias;

        private readonly Tensor __feats_2_smfa_lde_conv_0_0_weight;

        private readonly Tensor __feats_2_smfa_lde_conv_0_0_bias;

        private readonly Tensor __feats_2_smfa_lde_conv_0_1_weight;

        private readonly Tensor __feats_2_smfa_lde_conv_0_1_bias;

        private readonly Tensor __feats_2_smfa_lde_conv_1_weight;

        private readonly Tensor __feats_2_smfa_lde_conv_1_bias;

        private readonly Tensor __feats_2_smfa_dw_conv_weight;

        private readonly Tensor __feats_2_smfa_dw_conv_bias;

        private readonly Tensor __feats_2_pcfn_conv_0_weight;

        private readonly Tensor __feats_2_pcfn_conv_0_bias;

        private readonly Tensor __feats_2_pcfn_conv_1_weight;

        private readonly Tensor __feats_2_pcfn_conv_1_bias;

        private readonly Tensor __feats_2_pcfn_conv_2_weight;

        private readonly Tensor __feats_2_pcfn_conv_2_bias;

        private readonly Tensor __feats_3_smfa_alpha;

        private readonly Tensor __feats_3_smfa_belt;

        private readonly Tensor __feats_3_smfa_linear_0_weight;

        private readonly Tensor __feats_3_smfa_linear_0_bias;

        private readonly Tensor __feats_3_smfa_linear_1_weight;

        private readonly Tensor __feats_3_smfa_linear_1_bias;

        private readonly Tensor __feats_3_smfa_linear_2_weight;

        private readonly Tensor __feats_3_smfa_linear_2_bias;

        private readonly Tensor __feats_3_smfa_lde_conv_0_0_weight;

        private readonly Tensor __feats_3_smfa_lde_conv_0_0_bias;

        private readonly Tensor __feats_3_smfa_lde_conv_0_1_weight;

        private readonly Tensor __feats_3_smfa_lde_conv_0_1_bias;

        private readonly Tensor __feats_3_smfa_lde_conv_1_weight;

        private readonly Tensor __feats_3_smfa_lde_conv_1_bias;

        private readonly Tensor __feats_3_smfa_dw_conv_weight;

        private readonly Tensor __feats_3_smfa_dw_conv_bias;

        private readonly Tensor __feats_3_pcfn_conv_0_weight;

        private readonly Tensor __feats_3_pcfn_conv_0_bias;

        private readonly Tensor __feats_3_pcfn_conv_1_weight;

        private readonly Tensor __feats_3_pcfn_conv_1_bias;

        private readonly Tensor __feats_3_pcfn_conv_2_weight;

        private readonly Tensor __feats_3_pcfn_conv_2_bias;

        private readonly Tensor __feats_4_smfa_alpha;

        private readonly Tensor __feats_4_smfa_belt;

        private readonly Tensor __feats_4_smfa_linear_0_weight;

        private readonly Tensor __feats_4_smfa_linear_0_bias;

        private readonly Tensor __feats_4_smfa_linear_1_weight;

        private readonly Tensor __feats_4_smfa_linear_1_bias;

        private readonly Tensor __feats_4_smfa_linear_2_weight;

        private readonly Tensor __feats_4_smfa_linear_2_bias;

        private readonly Tensor __feats_4_smfa_lde_conv_0_0_weight;

        private readonly Tensor __feats_4_smfa_lde_conv_0_0_bias;

        private readonly Tensor __feats_4_smfa_lde_conv_0_1_weight;

        private readonly Tensor __feats_4_smfa_lde_conv_0_1_bias;

        private readonly Tensor __feats_4_smfa_lde_conv_1_weight;

        private readonly Tensor __feats_4_smfa_lde_conv_1_bias;

        private readonly Tensor __feats_4_smfa_dw_conv_weight;

        private readonly Tensor __feats_4_smfa_dw_conv_bias;

        private readonly Tensor __feats_4_pcfn_conv_0_weight;

        private readonly Tensor __feats_4_pcfn_conv_0_bias;

        private readonly Tensor __feats_4_pcfn_conv_1_weight;

        private readonly Tensor __feats_4_pcfn_conv_1_bias;

        private readonly Tensor __feats_4_pcfn_conv_2_weight;

        private readonly Tensor __feats_4_pcfn_conv_2_bias;

        private readonly Tensor __feats_5_smfa_alpha;

        private readonly Tensor __feats_5_smfa_belt;

        private readonly Tensor __feats_5_smfa_linear_0_weight;

        private readonly Tensor __feats_5_smfa_linear_0_bias;

        private readonly Tensor __feats_5_smfa_linear_1_weight;

        private readonly Tensor __feats_5_smfa_linear_1_bias;

        private readonly Tensor __feats_5_smfa_linear_2_weight;

        private readonly Tensor __feats_5_smfa_linear_2_bias;

        private readonly Tensor __feats_5_smfa_lde_conv_0_0_weight;

        private readonly Tensor __feats_5_smfa_lde_conv_0_0_bias;

        private readonly Tensor __feats_5_smfa_lde_conv_0_1_weight;

        private readonly Tensor __feats_5_smfa_lde_conv_0_1_bias;

        private readonly Tensor __feats_5_smfa_lde_conv_1_weight;

        private readonly Tensor __feats_5_smfa_lde_conv_1_bias;

        private readonly Tensor __feats_5_smfa_dw_conv_weight;

        private readonly Tensor __feats_5_smfa_dw_conv_bias;

        private readonly Tensor __feats_5_pcfn_conv_0_weight;

        private readonly Tensor __feats_5_pcfn_conv_0_bias;

        private readonly Tensor __feats_5_pcfn_conv_1_weight;

        private readonly Tensor __feats_5_pcfn_conv_1_bias;

        private readonly Tensor __feats_5_pcfn_conv_2_weight;

        private readonly Tensor __feats_5_pcfn_conv_2_bias;

        private readonly Tensor __feats_6_smfa_alpha;

        private readonly Tensor __feats_6_smfa_belt;

        private readonly Tensor __feats_6_smfa_linear_0_weight;

        private readonly Tensor __feats_6_smfa_linear_0_bias;

        private readonly Tensor __feats_6_smfa_linear_1_weight;

        private readonly Tensor __feats_6_smfa_linear_1_bias;

        private readonly Tensor __feats_6_smfa_linear_2_weight;

        private readonly Tensor __feats_6_smfa_linear_2_bias;

        private readonly Tensor __feats_6_smfa_lde_conv_0_0_weight;

        private readonly Tensor __feats_6_smfa_lde_conv_0_0_bias;

        private readonly Tensor __feats_6_smfa_lde_conv_0_1_weight;

        private readonly Tensor __feats_6_smfa_lde_conv_0_1_bias;

        private readonly Tensor __feats_6_smfa_lde_conv_1_weight;

        private readonly Tensor __feats_6_smfa_lde_conv_1_bias;

        private readonly Tensor __feats_6_smfa_dw_conv_weight;

        private readonly Tensor __feats_6_smfa_dw_conv_bias;

        private readonly Tensor __feats_6_pcfn_conv_0_weight;

        private readonly Tensor __feats_6_pcfn_conv_0_bias;

        private readonly Tensor __feats_6_pcfn_conv_1_weight;

        private readonly Tensor __feats_6_pcfn_conv_1_bias;

        private readonly Tensor __feats_6_pcfn_conv_2_weight;

        private readonly Tensor __feats_6_pcfn_conv_2_bias;

        private readonly Tensor __feats_7_smfa_alpha;

        private readonly Tensor __feats_7_smfa_belt;

        private readonly Tensor __feats_7_smfa_linear_0_weight;

        private readonly Tensor __feats_7_smfa_linear_0_bias;

        private readonly Tensor __feats_7_smfa_linear_1_weight;

        private readonly Tensor __feats_7_smfa_linear_1_bias;

        private readonly Tensor __feats_7_smfa_linear_2_weight;

        private readonly Tensor __feats_7_smfa_linear_2_bias;

        private readonly Tensor __feats_7_smfa_lde_conv_0_0_weight;

        private readonly Tensor __feats_7_smfa_lde_conv_0_0_bias;

        private readonly Tensor __feats_7_smfa_lde_conv_0_1_weight;

        private readonly Tensor __feats_7_smfa_lde_conv_0_1_bias;

        private readonly Tensor __feats_7_smfa_lde_conv_1_weight;

        private readonly Tensor __feats_7_smfa_lde_conv_1_bias;

        private readonly Tensor __feats_7_smfa_dw_conv_weight;

        private readonly Tensor __feats_7_smfa_dw_conv_bias;

        private readonly Tensor __feats_7_pcfn_conv_0_weight;

        private readonly Tensor __feats_7_pcfn_conv_0_bias;

        private readonly Tensor __feats_7_pcfn_conv_1_weight;

        private readonly Tensor __feats_7_pcfn_conv_1_bias;

        private readonly Tensor __feats_7_pcfn_conv_2_weight;

        private readonly Tensor __feats_7_pcfn_conv_2_bias;

        private readonly Tensor __feats_8_smfa_alpha;

        private readonly Tensor __feats_8_smfa_belt;

        private readonly Tensor __feats_8_smfa_linear_0_weight;

        private readonly Tensor __feats_8_smfa_linear_0_bias;

        private readonly Tensor __feats_8_smfa_linear_1_weight;

        private readonly Tensor __feats_8_smfa_linear_1_bias;

        private readonly Tensor __feats_8_smfa_linear_2_weight;

        private readonly Tensor __feats_8_smfa_linear_2_bias;

        private readonly Tensor __feats_8_smfa_lde_conv_0_0_weight;

        private readonly Tensor __feats_8_smfa_lde_conv_0_0_bias;

        private readonly Tensor __feats_8_smfa_lde_conv_0_1_weight;

        private readonly Tensor __feats_8_smfa_lde_conv_0_1_bias;

        private readonly Tensor __feats_8_smfa_lde_conv_1_weight;

        private readonly Tensor __feats_8_smfa_lde_conv_1_bias;

        private readonly Tensor __feats_8_smfa_dw_conv_weight;

        private readonly Tensor __feats_8_smfa_dw_conv_bias;

        private readonly Tensor __feats_8_pcfn_conv_0_weight;

        private readonly Tensor __feats_8_pcfn_conv_0_bias;

        private readonly Tensor __feats_8_pcfn_conv_1_weight;

        private readonly Tensor __feats_8_pcfn_conv_1_bias;

        private readonly Tensor __feats_8_pcfn_conv_2_weight;

        private readonly Tensor __feats_8_pcfn_conv_2_bias;

        private readonly Tensor __feats_9_smfa_alpha;

        private readonly Tensor __feats_9_smfa_belt;

        private readonly Tensor __feats_9_smfa_linear_0_weight;

        private readonly Tensor __feats_9_smfa_linear_0_bias;

        private readonly Tensor __feats_9_smfa_linear_1_weight;

        private readonly Tensor __feats_9_smfa_linear_1_bias;

        private readonly Tensor __feats_9_smfa_linear_2_weight;

        private readonly Tensor __feats_9_smfa_linear_2_bias;

        private readonly Tensor __feats_9_smfa_lde_conv_0_0_weight;

        private readonly Tensor __feats_9_smfa_lde_conv_0_0_bias;

        private readonly Tensor __feats_9_smfa_lde_conv_0_1_weight;

        private readonly Tensor __feats_9_smfa_lde_conv_0_1_bias;

        private readonly Tensor __feats_9_smfa_lde_conv_1_weight;

        private readonly Tensor __feats_9_smfa_lde_conv_1_bias;

        private readonly Tensor __feats_9_smfa_dw_conv_weight;

        private readonly Tensor __feats_9_smfa_dw_conv_bias;

        private readonly Tensor __feats_9_pcfn_conv_0_weight;

        private readonly Tensor __feats_9_pcfn_conv_0_bias;

        private readonly Tensor __feats_9_pcfn_conv_1_weight;

        private readonly Tensor __feats_9_pcfn_conv_1_bias;

        private readonly Tensor __feats_9_pcfn_conv_2_weight;

        private readonly Tensor __feats_9_pcfn_conv_2_bias;

        private readonly Tensor __feats_10_smfa_alpha;

        private readonly Tensor __feats_10_smfa_belt;

        private readonly Tensor __feats_10_smfa_linear_0_weight;

        private readonly Tensor __feats_10_smfa_linear_0_bias;

        private readonly Tensor __feats_10_smfa_linear_1_weight;

        private readonly Tensor __feats_10_smfa_linear_1_bias;

        private readonly Tensor __feats_10_smfa_linear_2_weight;

        private readonly Tensor __feats_10_smfa_linear_2_bias;

        private readonly Tensor __feats_10_smfa_lde_conv_0_0_weight;

        private readonly Tensor __feats_10_smfa_lde_conv_0_0_bias;

        private readonly Tensor __feats_10_smfa_lde_conv_0_1_weight;

        private readonly Tensor __feats_10_smfa_lde_conv_0_1_bias;

        private readonly Tensor __feats_10_smfa_lde_conv_1_weight;

        private readonly Tensor __feats_10_smfa_lde_conv_1_bias;

        private readonly Tensor __feats_10_smfa_dw_conv_weight;

        private readonly Tensor __feats_10_smfa_dw_conv_bias;

        private readonly Tensor __feats_10_pcfn_conv_0_weight;

        private readonly Tensor __feats_10_pcfn_conv_0_bias;

        private readonly Tensor __feats_10_pcfn_conv_1_weight;

        private readonly Tensor __feats_10_pcfn_conv_1_bias;

        private readonly Tensor __feats_10_pcfn_conv_2_weight;

        private readonly Tensor __feats_10_pcfn_conv_2_bias;

        private readonly Tensor __feats_11_smfa_alpha;

        private readonly Tensor __feats_11_smfa_belt;

        private readonly Tensor __feats_11_smfa_linear_0_weight;

        private readonly Tensor __feats_11_smfa_linear_0_bias;

        private readonly Tensor __feats_11_smfa_linear_1_weight;

        private readonly Tensor __feats_11_smfa_linear_1_bias;

        private readonly Tensor __feats_11_smfa_linear_2_weight;

        private readonly Tensor __feats_11_smfa_linear_2_bias;

        private readonly Tensor __feats_11_smfa_lde_conv_0_0_weight;

        private readonly Tensor __feats_11_smfa_lde_conv_0_0_bias;

        private readonly Tensor __feats_11_smfa_lde_conv_0_1_weight;

        private readonly Tensor __feats_11_smfa_lde_conv_0_1_bias;

        private readonly Tensor __feats_11_smfa_lde_conv_1_weight;

        private readonly Tensor __feats_11_smfa_lde_conv_1_bias;

        private readonly Tensor __feats_11_smfa_dw_conv_weight;

        private readonly Tensor __feats_11_smfa_dw_conv_bias;

        private readonly Tensor __feats_11_pcfn_conv_0_weight;

        private readonly Tensor __feats_11_pcfn_conv_0_bias;

        private readonly Tensor __feats_11_pcfn_conv_1_weight;

        private readonly Tensor __feats_11_pcfn_conv_1_bias;

        private readonly Tensor __feats_11_pcfn_conv_2_weight;

        private readonly Tensor __feats_11_pcfn_conv_2_bias;

        private readonly Tensor __to_img_0_weight;

        private readonly Tensor __to_img_0_bias;

        private SMFANet()
        {
            BinaryReader reader = new BinaryReader(Assembly.GetExecutingAssembly().GetManifestResourceStream("SMFANet.hmodel"));
            __load_parameter(reader, ref this.__to_feat_weight, 48, 3, 3, 3);
            __load_parameter(reader, ref this.__to_feat_bias, 48);
            __load_parameter(reader, ref this.__feats_0_smfa_alpha, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_0_smfa_belt, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_0_smfa_linear_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_0_smfa_linear_0_bias, 96);
            __load_parameter(reader, ref this.__feats_0_smfa_linear_1_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_0_smfa_linear_1_bias, 48);
            __load_parameter(reader, ref this.__feats_0_smfa_linear_2_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_0_smfa_linear_2_bias, 48);
            __load_parameter(reader, ref this.__feats_0_smfa_lde_conv_0_0_weight, 96, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_0_smfa_lde_conv_0_0_bias, 96);
            __load_parameter(reader, ref this.__feats_0_smfa_lde_conv_0_1_weight, 96, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_0_smfa_lde_conv_0_1_bias, 96);
            __load_parameter(reader, ref this.__feats_0_smfa_lde_conv_1_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_0_smfa_lde_conv_1_bias, 48);
            __load_parameter(reader, ref this.__feats_0_smfa_dw_conv_weight, 48, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_0_smfa_dw_conv_bias, 48);
            __load_parameter(reader, ref this.__feats_0_pcfn_conv_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_0_pcfn_conv_0_bias, 96);
            __load_parameter(reader, ref this.__feats_0_pcfn_conv_1_weight, 24, 24, 3, 3);
            __load_parameter(reader, ref this.__feats_0_pcfn_conv_1_bias, 24);
            __load_parameter(reader, ref this.__feats_0_pcfn_conv_2_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_0_pcfn_conv_2_bias, 48);
            __load_parameter(reader, ref this.__feats_1_smfa_alpha, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_1_smfa_belt, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_1_smfa_linear_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_1_smfa_linear_0_bias, 96);
            __load_parameter(reader, ref this.__feats_1_smfa_linear_1_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_1_smfa_linear_1_bias, 48);
            __load_parameter(reader, ref this.__feats_1_smfa_linear_2_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_1_smfa_linear_2_bias, 48);
            __load_parameter(reader, ref this.__feats_1_smfa_lde_conv_0_0_weight, 96, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_1_smfa_lde_conv_0_0_bias, 96);
            __load_parameter(reader, ref this.__feats_1_smfa_lde_conv_0_1_weight, 96, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_1_smfa_lde_conv_0_1_bias, 96);
            __load_parameter(reader, ref this.__feats_1_smfa_lde_conv_1_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_1_smfa_lde_conv_1_bias, 48);
            __load_parameter(reader, ref this.__feats_1_smfa_dw_conv_weight, 48, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_1_smfa_dw_conv_bias, 48);
            __load_parameter(reader, ref this.__feats_1_pcfn_conv_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_1_pcfn_conv_0_bias, 96);
            __load_parameter(reader, ref this.__feats_1_pcfn_conv_1_weight, 24, 24, 3, 3);
            __load_parameter(reader, ref this.__feats_1_pcfn_conv_1_bias, 24);
            __load_parameter(reader, ref this.__feats_1_pcfn_conv_2_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_1_pcfn_conv_2_bias, 48);
            __load_parameter(reader, ref this.__feats_2_smfa_alpha, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_2_smfa_belt, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_2_smfa_linear_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_2_smfa_linear_0_bias, 96);
            __load_parameter(reader, ref this.__feats_2_smfa_linear_1_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_2_smfa_linear_1_bias, 48);
            __load_parameter(reader, ref this.__feats_2_smfa_linear_2_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_2_smfa_linear_2_bias, 48);
            __load_parameter(reader, ref this.__feats_2_smfa_lde_conv_0_0_weight, 96, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_2_smfa_lde_conv_0_0_bias, 96);
            __load_parameter(reader, ref this.__feats_2_smfa_lde_conv_0_1_weight, 96, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_2_smfa_lde_conv_0_1_bias, 96);
            __load_parameter(reader, ref this.__feats_2_smfa_lde_conv_1_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_2_smfa_lde_conv_1_bias, 48);
            __load_parameter(reader, ref this.__feats_2_smfa_dw_conv_weight, 48, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_2_smfa_dw_conv_bias, 48);
            __load_parameter(reader, ref this.__feats_2_pcfn_conv_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_2_pcfn_conv_0_bias, 96);
            __load_parameter(reader, ref this.__feats_2_pcfn_conv_1_weight, 24, 24, 3, 3);
            __load_parameter(reader, ref this.__feats_2_pcfn_conv_1_bias, 24);
            __load_parameter(reader, ref this.__feats_2_pcfn_conv_2_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_2_pcfn_conv_2_bias, 48);
            __load_parameter(reader, ref this.__feats_3_smfa_alpha, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_3_smfa_belt, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_3_smfa_linear_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_3_smfa_linear_0_bias, 96);
            __load_parameter(reader, ref this.__feats_3_smfa_linear_1_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_3_smfa_linear_1_bias, 48);
            __load_parameter(reader, ref this.__feats_3_smfa_linear_2_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_3_smfa_linear_2_bias, 48);
            __load_parameter(reader, ref this.__feats_3_smfa_lde_conv_0_0_weight, 96, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_3_smfa_lde_conv_0_0_bias, 96);
            __load_parameter(reader, ref this.__feats_3_smfa_lde_conv_0_1_weight, 96, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_3_smfa_lde_conv_0_1_bias, 96);
            __load_parameter(reader, ref this.__feats_3_smfa_lde_conv_1_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_3_smfa_lde_conv_1_bias, 48);
            __load_parameter(reader, ref this.__feats_3_smfa_dw_conv_weight, 48, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_3_smfa_dw_conv_bias, 48);
            __load_parameter(reader, ref this.__feats_3_pcfn_conv_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_3_pcfn_conv_0_bias, 96);
            __load_parameter(reader, ref this.__feats_3_pcfn_conv_1_weight, 24, 24, 3, 3);
            __load_parameter(reader, ref this.__feats_3_pcfn_conv_1_bias, 24);
            __load_parameter(reader, ref this.__feats_3_pcfn_conv_2_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_3_pcfn_conv_2_bias, 48);
            __load_parameter(reader, ref this.__feats_4_smfa_alpha, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_4_smfa_belt, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_4_smfa_linear_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_4_smfa_linear_0_bias, 96);
            __load_parameter(reader, ref this.__feats_4_smfa_linear_1_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_4_smfa_linear_1_bias, 48);
            __load_parameter(reader, ref this.__feats_4_smfa_linear_2_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_4_smfa_linear_2_bias, 48);
            __load_parameter(reader, ref this.__feats_4_smfa_lde_conv_0_0_weight, 96, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_4_smfa_lde_conv_0_0_bias, 96);
            __load_parameter(reader, ref this.__feats_4_smfa_lde_conv_0_1_weight, 96, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_4_smfa_lde_conv_0_1_bias, 96);
            __load_parameter(reader, ref this.__feats_4_smfa_lde_conv_1_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_4_smfa_lde_conv_1_bias, 48);
            __load_parameter(reader, ref this.__feats_4_smfa_dw_conv_weight, 48, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_4_smfa_dw_conv_bias, 48);
            __load_parameter(reader, ref this.__feats_4_pcfn_conv_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_4_pcfn_conv_0_bias, 96);
            __load_parameter(reader, ref this.__feats_4_pcfn_conv_1_weight, 24, 24, 3, 3);
            __load_parameter(reader, ref this.__feats_4_pcfn_conv_1_bias, 24);
            __load_parameter(reader, ref this.__feats_4_pcfn_conv_2_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_4_pcfn_conv_2_bias, 48);
            __load_parameter(reader, ref this.__feats_5_smfa_alpha, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_5_smfa_belt, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_5_smfa_linear_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_5_smfa_linear_0_bias, 96);
            __load_parameter(reader, ref this.__feats_5_smfa_linear_1_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_5_smfa_linear_1_bias, 48);
            __load_parameter(reader, ref this.__feats_5_smfa_linear_2_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_5_smfa_linear_2_bias, 48);
            __load_parameter(reader, ref this.__feats_5_smfa_lde_conv_0_0_weight, 96, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_5_smfa_lde_conv_0_0_bias, 96);
            __load_parameter(reader, ref this.__feats_5_smfa_lde_conv_0_1_weight, 96, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_5_smfa_lde_conv_0_1_bias, 96);
            __load_parameter(reader, ref this.__feats_5_smfa_lde_conv_1_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_5_smfa_lde_conv_1_bias, 48);
            __load_parameter(reader, ref this.__feats_5_smfa_dw_conv_weight, 48, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_5_smfa_dw_conv_bias, 48);
            __load_parameter(reader, ref this.__feats_5_pcfn_conv_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_5_pcfn_conv_0_bias, 96);
            __load_parameter(reader, ref this.__feats_5_pcfn_conv_1_weight, 24, 24, 3, 3);
            __load_parameter(reader, ref this.__feats_5_pcfn_conv_1_bias, 24);
            __load_parameter(reader, ref this.__feats_5_pcfn_conv_2_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_5_pcfn_conv_2_bias, 48);
            __load_parameter(reader, ref this.__feats_6_smfa_alpha, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_6_smfa_belt, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_6_smfa_linear_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_6_smfa_linear_0_bias, 96);
            __load_parameter(reader, ref this.__feats_6_smfa_linear_1_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_6_smfa_linear_1_bias, 48);
            __load_parameter(reader, ref this.__feats_6_smfa_linear_2_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_6_smfa_linear_2_bias, 48);
            __load_parameter(reader, ref this.__feats_6_smfa_lde_conv_0_0_weight, 96, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_6_smfa_lde_conv_0_0_bias, 96);
            __load_parameter(reader, ref this.__feats_6_smfa_lde_conv_0_1_weight, 96, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_6_smfa_lde_conv_0_1_bias, 96);
            __load_parameter(reader, ref this.__feats_6_smfa_lde_conv_1_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_6_smfa_lde_conv_1_bias, 48);
            __load_parameter(reader, ref this.__feats_6_smfa_dw_conv_weight, 48, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_6_smfa_dw_conv_bias, 48);
            __load_parameter(reader, ref this.__feats_6_pcfn_conv_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_6_pcfn_conv_0_bias, 96);
            __load_parameter(reader, ref this.__feats_6_pcfn_conv_1_weight, 24, 24, 3, 3);
            __load_parameter(reader, ref this.__feats_6_pcfn_conv_1_bias, 24);
            __load_parameter(reader, ref this.__feats_6_pcfn_conv_2_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_6_pcfn_conv_2_bias, 48);
            __load_parameter(reader, ref this.__feats_7_smfa_alpha, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_7_smfa_belt, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_7_smfa_linear_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_7_smfa_linear_0_bias, 96);
            __load_parameter(reader, ref this.__feats_7_smfa_linear_1_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_7_smfa_linear_1_bias, 48);
            __load_parameter(reader, ref this.__feats_7_smfa_linear_2_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_7_smfa_linear_2_bias, 48);
            __load_parameter(reader, ref this.__feats_7_smfa_lde_conv_0_0_weight, 96, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_7_smfa_lde_conv_0_0_bias, 96);
            __load_parameter(reader, ref this.__feats_7_smfa_lde_conv_0_1_weight, 96, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_7_smfa_lde_conv_0_1_bias, 96);
            __load_parameter(reader, ref this.__feats_7_smfa_lde_conv_1_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_7_smfa_lde_conv_1_bias, 48);
            __load_parameter(reader, ref this.__feats_7_smfa_dw_conv_weight, 48, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_7_smfa_dw_conv_bias, 48);
            __load_parameter(reader, ref this.__feats_7_pcfn_conv_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_7_pcfn_conv_0_bias, 96);
            __load_parameter(reader, ref this.__feats_7_pcfn_conv_1_weight, 24, 24, 3, 3);
            __load_parameter(reader, ref this.__feats_7_pcfn_conv_1_bias, 24);
            __load_parameter(reader, ref this.__feats_7_pcfn_conv_2_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_7_pcfn_conv_2_bias, 48);
            __load_parameter(reader, ref this.__feats_8_smfa_alpha, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_8_smfa_belt, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_8_smfa_linear_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_8_smfa_linear_0_bias, 96);
            __load_parameter(reader, ref this.__feats_8_smfa_linear_1_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_8_smfa_linear_1_bias, 48);
            __load_parameter(reader, ref this.__feats_8_smfa_linear_2_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_8_smfa_linear_2_bias, 48);
            __load_parameter(reader, ref this.__feats_8_smfa_lde_conv_0_0_weight, 96, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_8_smfa_lde_conv_0_0_bias, 96);
            __load_parameter(reader, ref this.__feats_8_smfa_lde_conv_0_1_weight, 96, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_8_smfa_lde_conv_0_1_bias, 96);
            __load_parameter(reader, ref this.__feats_8_smfa_lde_conv_1_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_8_smfa_lde_conv_1_bias, 48);
            __load_parameter(reader, ref this.__feats_8_smfa_dw_conv_weight, 48, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_8_smfa_dw_conv_bias, 48);
            __load_parameter(reader, ref this.__feats_8_pcfn_conv_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_8_pcfn_conv_0_bias, 96);
            __load_parameter(reader, ref this.__feats_8_pcfn_conv_1_weight, 24, 24, 3, 3);
            __load_parameter(reader, ref this.__feats_8_pcfn_conv_1_bias, 24);
            __load_parameter(reader, ref this.__feats_8_pcfn_conv_2_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_8_pcfn_conv_2_bias, 48);
            __load_parameter(reader, ref this.__feats_9_smfa_alpha, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_9_smfa_belt, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_9_smfa_linear_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_9_smfa_linear_0_bias, 96);
            __load_parameter(reader, ref this.__feats_9_smfa_linear_1_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_9_smfa_linear_1_bias, 48);
            __load_parameter(reader, ref this.__feats_9_smfa_linear_2_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_9_smfa_linear_2_bias, 48);
            __load_parameter(reader, ref this.__feats_9_smfa_lde_conv_0_0_weight, 96, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_9_smfa_lde_conv_0_0_bias, 96);
            __load_parameter(reader, ref this.__feats_9_smfa_lde_conv_0_1_weight, 96, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_9_smfa_lde_conv_0_1_bias, 96);
            __load_parameter(reader, ref this.__feats_9_smfa_lde_conv_1_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_9_smfa_lde_conv_1_bias, 48);
            __load_parameter(reader, ref this.__feats_9_smfa_dw_conv_weight, 48, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_9_smfa_dw_conv_bias, 48);
            __load_parameter(reader, ref this.__feats_9_pcfn_conv_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_9_pcfn_conv_0_bias, 96);
            __load_parameter(reader, ref this.__feats_9_pcfn_conv_1_weight, 24, 24, 3, 3);
            __load_parameter(reader, ref this.__feats_9_pcfn_conv_1_bias, 24);
            __load_parameter(reader, ref this.__feats_9_pcfn_conv_2_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_9_pcfn_conv_2_bias, 48);
            __load_parameter(reader, ref this.__feats_10_smfa_alpha, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_10_smfa_belt, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_10_smfa_linear_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_10_smfa_linear_0_bias, 96);
            __load_parameter(reader, ref this.__feats_10_smfa_linear_1_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_10_smfa_linear_1_bias, 48);
            __load_parameter(reader, ref this.__feats_10_smfa_linear_2_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_10_smfa_linear_2_bias, 48);
            __load_parameter(reader, ref this.__feats_10_smfa_lde_conv_0_0_weight, 96, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_10_smfa_lde_conv_0_0_bias, 96);
            __load_parameter(reader, ref this.__feats_10_smfa_lde_conv_0_1_weight, 96, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_10_smfa_lde_conv_0_1_bias, 96);
            __load_parameter(reader, ref this.__feats_10_smfa_lde_conv_1_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_10_smfa_lde_conv_1_bias, 48);
            __load_parameter(reader, ref this.__feats_10_smfa_dw_conv_weight, 48, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_10_smfa_dw_conv_bias, 48);
            __load_parameter(reader, ref this.__feats_10_pcfn_conv_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_10_pcfn_conv_0_bias, 96);
            __load_parameter(reader, ref this.__feats_10_pcfn_conv_1_weight, 24, 24, 3, 3);
            __load_parameter(reader, ref this.__feats_10_pcfn_conv_1_bias, 24);
            __load_parameter(reader, ref this.__feats_10_pcfn_conv_2_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_10_pcfn_conv_2_bias, 48);
            __load_parameter(reader, ref this.__feats_11_smfa_alpha, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_11_smfa_belt, 1, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_11_smfa_linear_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_11_smfa_linear_0_bias, 96);
            __load_parameter(reader, ref this.__feats_11_smfa_linear_1_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_11_smfa_linear_1_bias, 48);
            __load_parameter(reader, ref this.__feats_11_smfa_linear_2_weight, 48, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_11_smfa_linear_2_bias, 48);
            __load_parameter(reader, ref this.__feats_11_smfa_lde_conv_0_0_weight, 96, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_11_smfa_lde_conv_0_0_bias, 96);
            __load_parameter(reader, ref this.__feats_11_smfa_lde_conv_0_1_weight, 96, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_11_smfa_lde_conv_0_1_bias, 96);
            __load_parameter(reader, ref this.__feats_11_smfa_lde_conv_1_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_11_smfa_lde_conv_1_bias, 48);
            __load_parameter(reader, ref this.__feats_11_smfa_dw_conv_weight, 48, 1, 3, 3);
            __load_parameter(reader, ref this.__feats_11_smfa_dw_conv_bias, 48);
            __load_parameter(reader, ref this.__feats_11_pcfn_conv_0_weight, 96, 48, 1, 1);
            __load_parameter(reader, ref this.__feats_11_pcfn_conv_0_bias, 96);
            __load_parameter(reader, ref this.__feats_11_pcfn_conv_1_weight, 24, 24, 3, 3);
            __load_parameter(reader, ref this.__feats_11_pcfn_conv_1_bias, 24);
            __load_parameter(reader, ref this.__feats_11_pcfn_conv_2_weight, 48, 96, 1, 1);
            __load_parameter(reader, ref this.__feats_11_pcfn_conv_2_bias, 48);
            __load_parameter(reader, ref this.__to_img_0_weight, 48, 48, 3, 3);
            __load_parameter(reader, ref this.__to_img_0_bias, 48);
        }

        private static float __half2float(short float16)
        {
            int sign = float16 >> 15;
            int exponent = (float16 >> 10) & 0x1F;
            int fraction = (float16 & 0x3FF);
            int float32;
            if(exponent == 0)
            {
                if(fraction == 0)
                {
                    float32 = (sign << 31);
                }
                else
                {
                    exponent = 127 - 14;
                    while((fraction & (1 << 10)) == 0)
                    {
                        exponent--;
                        fraction <<= 1;
                    }
                    fraction &= 0x3FF;
                    float32 = (sign << 31) | (exponent << 23) | (fraction << 13);
                }
            }
            else
            {
                if(exponent == 0x1F)
                {
                    float32 = (sign << 31) | (0xFF << 23) | (fraction << 13);
                }
                else
                {
                    float32 = (sign << 31) | ((exponent + (127 - 15)) << 23) | (fraction << 13);
                }
            }
            return *((float*)&float32);
        }

        private static void __load_parameter(BinaryReader reader, ref Tensor dst, params int[] shape)
        {
            dst = new Tensor(shape);
            float* dst_ptr = dst.data_ptr;
            int numel = dst.numel;
            for(int i = 0; i < numel; ++i)
            {
                dst_ptr[i] = __half2float(reader.ReadInt16());
            }
        }

        private static Tensor __bitmap2tensor(Bitmap bmp)
        {
            int height = bmp.Height;
            int width = bmp.Width;
            Tensor t = new Tensor(3, height, width);
            float* dst = t.data_ptr;
            BitmapData bd = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            for(int y = 0; y < height; ++y)
            {
                byte* curpos = ((byte*)bd.Scan0) + y * bd.Stride;
                for(int x = 0; x < width; ++x)
                {
                    dst[(2 * height + y) * width + x] = *(curpos++) / 255f;
                    dst[(1 * height + y) * width + x] = *(curpos++) / 255f;
                    dst[(0 * height + y) * width + x] = *(curpos++) / 255f;
                }
            }
            bmp.UnlockBits(bd);
            return t;
        }

        private static float __clamp(float value, float min, float max)
        {
            if(value < min)
            {
                return min;
            }
            if(value > max)
            {
                return max;
            }
            return value;
        }

        private static Bitmap __tensor2bitmap(Tensor t)
        {
            int height = t.shape_ptr[1];
            int width = t.shape_ptr[2];
            Bitmap bmp = new Bitmap(width, height);
            float* src = t.data_ptr;
            BitmapData bd = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
            for(int y = 0; y < height; ++y)
            {
                byte* curpos = ((byte*)bd.Scan0) + y * bd.Stride;
                for(int x = 0; x < width; ++x)
                {
                    *(curpos++) = (byte)__clamp(src[(2 * height + y) * width + x] * 255, 0, 255);
                    *(curpos++) = (byte)__clamp(src[(1 * height + y) * width + x] * 255, 0, 255);
                    *(curpos++) = (byte)__clamp(src[(0 * height + y) * width + x] * 255, 0, 255);
                }
            }
            GC.KeepAlive(t);
            bmp.UnlockBits(bd);
            return bmp;
        }

        private static Tensor __conv2d_k3s1(Tensor input, Tensor weight, Tensor bias)
        {
            int height = input.shape_ptr[1];
            int width = input.shape_ptr[2];
            int src_c = input.shape_ptr[0];
            int dst_c = weight.shape_ptr[0];
            Tensor output = new Tensor(dst_c, height, width);
            float* src = input.data_ptr;
            float* w = weight.data_ptr;
            float* b = bias.data_ptr;
            float* dst = output.data_ptr;
            Parallel.For(0, height, (int dy) =>
            {
                float* buffer = stackalloc float[src_c * 9];
                float* dst1 = dst + dy * width;
                for(int dx = 0; dx < width; ++dx)
                {
                    float* dst2 = dst1 + dx;
                    for(int sc = 0; sc < src_c; ++sc)
                    {
                        float* src1 = src + sc * height * width;
                        for(int ky = -1; ky < 2; ++ky)
                        {
                            int sy = dy + ky;
                            if((sy < 0) || (sy >= height))
                            {
                                *buffer++ = 0f;
                                *buffer++ = 0f;
                                *buffer++ = 0f;
                                continue;
                            }
                            float* src2 = src1 + sy * width;
                            for(int kx = -1; kx < 2; ++kx)
                            {
                                int sx = dx + kx;
                                if((sx >= 0) && (sx < width))
                                {
                                    *buffer++ = src2[sx];
                                }
                                else
                                {
                                    *buffer++ = 0f;
                                }
                            }
                        }
                    }
                    buffer -= src_c * 9;
                    for(int dc = 0; dc < dst_c; ++dc)
                    {
                        float* w1 = w + dc * src_c * 9;
                        float sum = 0;
                        for(int n = 0; n < src_c * 9; ++n)
                        {
                            sum += buffer[n] * w1[n];
                        }
                        dst2[dc * height * width] = sum + b[dc];
                    }
                }
            });
            GC.KeepAlive(input);
            return output;
        }

        private static Tensor __residual_smfa_norm(Tensor input,
				                                   Tensor linear_0_weight,
				                                   Tensor linear_0_bias,
					                               Tensor linear_1_weight,
					                               Tensor linear_1_bias,
					                               Tensor linear_2_weight,
					                               Tensor linear_2_bias,
					                               Tensor lde_conv_0_0_weight,
					                               Tensor lde_conv_0_0_bias,
					                               Tensor lde_conv_0_1_weight,
					                               Tensor lde_conv_0_1_bias,
					                               Tensor lde_conv_1_weight,
					                               Tensor lde_conv_1_bias,
					                               Tensor dw_conv_weight,
					                               Tensor dw_conv_bias,
					                               Tensor alpha,
					                               Tensor belt)
        {
        	int src_c = input.shape_ptr[0];
        	int src_h = input.shape_ptr[1];
        	int src_w = input.shape_ptr[2];
        	float* src = input.data_ptr;
        	Tensor output = new Tensor(src_c, src_h, src_w);
        	float* dst = output.data_ptr;
        	// y, x = linear_0(input).chunk(2, dim=1)
        	float* l0_w = linear_0_weight.data_ptr;
        	float* l0_b = linear_0_bias.data_ptr;
        	float* y = (float*)Marshal.AllocCoTaskMem(src_c * src_h * src_w * sizeof(float));
        	float* x = (float*)Marshal.AllocCoTaskMem(src_c * src_h * src_w * sizeof(float));
        	Parallel.For(0, src_h, (int dy) =>
            {
        	    const float eps = 1e-12f;
        	    float* buffer = stackalloc float[src_c];
        		for(int dx = 0; dx < src_w; ++dx)
        		{
        			float tmp = 0f;
        			for(int sc = 0; sc < src_c; ++sc)
        			{
        				tmp += src[(sc * src_h + dy) * src_w + dx] * src[(sc * src_h + dy) * src_w + dx];
        			}
        			tmp = (float)Math.Sqrt(tmp);
        			if(tmp < eps)
        			{
        				tmp = eps;
        			}
        			for(int sc = 0; sc < src_c; ++sc)
        			{
        				buffer[sc] = src[(sc * src_h + dy) * src_w + dx] / tmp;
        			}
        			for(int dc = 0; dc < src_c; ++dc)
        			{
        				tmp = 0f;
        				for(int sc = 0; sc < src_c; ++sc)
        				{
        					tmp += buffer[sc] * l0_w[dc * src_c + sc];
        				}
        				y[(dc * src_h + dy) * src_w + dx] = tmp + l0_b[dc];
        			}
        			for(int dc = 0; dc < src_c; ++dc)
        			{
        				tmp = 0f;
        				for(int sc = 0; sc < src_c; ++sc)
        				{
        					tmp += buffer[sc] * l0_w[(src_c + dc) * src_c + sc];
        				}
        				x[(dc * src_h + dy) * src_w + dx] = tmp + l0_b[src_c + dc];
        			}
        		}
            });
        	// x_s = dw_conv(F.adaptive_max_pool2d(x, (h // down_scale, w // down_scale)))
        	// x_v = torch.var(x, dim=(-2,-1), keepdim=True)
        	// x_s = x_s * alpha + x_v * belt;
        	float* x_s = (float*)Marshal.AllocCoTaskMem(src_c * (src_h / 8) * (src_w / 8) * sizeof(float));
        	float* dw_w = dw_conv_weight.data_ptr;
        	float* dw_b = dw_conv_bias.data_ptr;
        	float* a = alpha.data_ptr;
        	float* b = belt.data_ptr;
        	Parallel.For(0, src_c, (int sc) =>
        	{
        	    float tmp = 0f;
        	    for(int sy = 0; sy < src_h; ++sy)
        	    {
        	    	for(int sx = 0; sx < src_w; ++sx)
        	    	{
        	    		tmp += x[(sc * src_h + sy) * src_w + sx];
        	    	}
        	    }
        	    tmp /= src_h * src_w;
        	    float x_v = 0f;
        	    for(int sy = 0; sy < src_h; ++sy)
        	    {
        	    	for(int sx = 0; sx < src_w; ++sx)
        	    	{
        	    		x_v += (x[(sc * src_h + sy) * src_w + sx] - tmp) * (x[(sc * src_h + sy) * src_w + sx] - tmp);
        	    	}
        	    }
        	    x_v /= src_h * src_w - 1;
        		float* buffer = stackalloc float[(src_h / 8) * (src_w / 8)];
        		for(int dy = 0; dy < src_h / 8; ++dy)
        		{
        			int sy0 = (dy / (src_h / 8)) * src_h + ((dy % (src_h / 8)) * src_h) / (src_h / 8);
        			int sy1 = 1 + ((dy + 1) * src_h - 1) / (src_h / 8);
        			for(int dx = 0; dx < src_w / 8; ++dx)
        			{
        				int sx0 = (dx / (src_w / 8)) * src_w + ((dx % (src_w / 8)) * src_w) / (src_w / 8);
        				int sx1 = 1 + ((dx + 1) * src_w - 1) / (src_w / 8);
        				float max = float.MinValue;
        				for(int sy = sy0; sy < sy1; ++sy)
        				{
        					for(int sx = sx0; sx < sx1; ++sx)
        					{
        						if(x[(sc * src_h + sy) * src_w + sx] > max)
        						{
        							max = x[(sc * src_h + sy) * src_w + sx];
        						}
        					}
        				}
        				buffer[dy * (src_w / 8) + dx] = max;
        			}
        		}
        		for(int dy = 0; dy < src_h / 8; ++dy)
        		{
        			for(int dx = 0; dx < src_w / 8; ++dx)
        			{
        				tmp = 0f;
        				for(int ky = 0; ky < 3; ++ky)
        				{
        					int sy = dy + ky - 1;
        					if((sy < 0) || (sy >= src_h / 8))
        					{
        						continue;
        					}
        					for(int kx = 0; kx < 3; ++kx)
        					{
        						int sx = dx + kx - 1;
	        					if((sx < 0) || (sx >= src_w / 8))
	        					{
	        						continue;
	        					}
	        					tmp += buffer[sy * (src_w / 8) + sx] * dw_w[(sc * 3 + ky) * 3 + kx];
        					}
        				}
        				x_s[(sc * (src_h / 8) + dy) * (src_w / 8) + dx] = (tmp + dw_b[sc]) * a[sc] + x_v * b[sc];
        			}
        		}
        	});
        	// x_s = GELU(linear_1(x_s))
        	float* l1_w = linear_1_weight.data_ptr;
        	float* l1_b = linear_1_bias.data_ptr;
        	Parallel.For(0, src_h / 8, (int dy) =>
        	{
        	    const float a1 = 0.254829592f;
        	    const float a2 = -0.284496736f;
        	    const float a3 = 1.421413741f;
        	    const float a4 = -1.453152027f;
        	    const float a5 = 1.061405429f;
        	    const float pc = 0.3275911f;
        	    const float sqrt2 = 1.41421356f;
        		float* buffer = stackalloc float[src_c];
        		for(int dx = 0; dx < src_w / 8; ++dx)
        		{
        			for(int sc = 0; sc < src_c; ++sc)
        			{
        				buffer[sc] = x_s[(sc * (src_h / 8) + dy) * (src_w / 8) + dx];
        			}
        			for(int dc = 0; dc < src_c; ++dc)
        			{
        				float tmp = 0f;
        				for(int sc = 0; sc < src_c; ++sc)
        				{
        					tmp += buffer[sc] * l1_w[dc * src_c + sc];
        				}
        				tmp += l1_b[dc];
        				// GELU
        				int sign;
        				float x_;
        				if(tmp < 0)
        				{
        					sign = -1;
        					x_ = -tmp / sqrt2;
        				}
        				else
        				{
        					sign = 1;
        					x_ = tmp / sqrt2;
        				}
        				float t = 1f / (1f + pc * x_);
        				float y_ = 1f - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (float)Math.Exp(-x_ * x_);
        				x_s[(dc * (src_h / 8) + dy) * (src_w / 8) + dx] = tmp * (sign * y_ + 1f) * 0.5f;
        			}
        		}
        	});
        	// x = x * F.interpolate(x_s, size=(h,w), mode='nearest')
        	Parallel.For(0, src_c, (int dc) =>
        	{
        		for(int dy = 0; dy < src_h; ++dy)
        		{
        			int sy = Math.Max(Math.Min((int)Math.Floor(dy * ((float)(src_h / 8) / src_h)), src_h - 1), 0);
        			for(int dx = 0; dx < src_w; ++dx)
        			{
        				int sx = Math.Max(Math.Min((int)Math.Floor(dx * ((float)(src_w / 8) / src_w)), src_w - 1), 0);
        				x[(dc * src_h + dy) * src_w + dx] *= x_s[(dc * (src_h / 8) + sy) * (src_w / 8) + sx];
        			}
        		}
        	});
        	Marshal.FreeCoTaskMem((IntPtr)x_s);
        	// y_d = lde(y)
        	// return linear_2(x + y_d) + input
        	int hidden_c = lde_conv_0_0_weight.shape_ptr[0];
        	float* lde0_w = lde_conv_0_0_weight.data_ptr;
        	float* lde0_b = lde_conv_0_0_bias.data_ptr;
        	float* lde1_w = lde_conv_0_1_weight.data_ptr;
        	float* lde1_b = lde_conv_0_1_bias.data_ptr;
        	float* lde2_w = lde_conv_1_weight.data_ptr;
        	float* lde2_b = lde_conv_1_bias.data_ptr;
        	float* l2_w = linear_2_weight.data_ptr;
        	float* l2_b = linear_2_bias.data_ptr;
        	Parallel.For(0, src_h, (int dy) =>
            {
        	    const float a1 = 0.254829592f;
        	    const float a2 = -0.284496736f;
        	    const float a3 = 1.421413741f;
        	    const float a4 = -1.453152027f;
        	    const float a5 = 1.061405429f;
        	    const float pc = 0.3275911f;
        	    const float sqrt2 = 1.41421356f;
        	    float* buffer0 = stackalloc float[hidden_c];
        	    float* buffer1 = stackalloc float[hidden_c];
        	    float* buffer2 = stackalloc float[src_c];
        	    float* buffer3 = stackalloc float[9];
        	    for(int dx = 0; dx < src_w; ++dx)
        	    {
        	    	for(int sc = 0; sc < src_c; ++sc)
        	    	{
        	    		for(int ky = 0; ky < 3; ++ky)
        	    		{
        	    			int sy = dy + ky - 1;
        	    			if((sy < 0) || (sy >= src_h))
        	    			{
        	    				*buffer3++ = 0f;
        	    				*buffer3++ = 0f;
        	    				*buffer3++ = 0f;
        	    				continue;
        	    			}
        	    			for(int kx = 0; kx < 3; ++kx)
        	    			{
        	    				int sx = dx + kx - 1;
	        	    			if((sx < 0) || (sx >= src_w))
	        	    			{
	        	    				*buffer3++ = 0f;
	        	    				continue;
	        	    			}
        	    				*buffer3++ = y[(sc * src_h + sy) * src_w + sx];
        	    			}
        	    		}
        	    		buffer3 -= 9;
        	    		for(int dc = 0; dc < hidden_c / src_c; ++dc)
        	    		{
        	    			float tmp = 0f;
        	    			for(int i = 0; i < 9; ++i)
        	    			{
        	    				tmp += buffer3[i] * lde0_w[(sc * 2 + dc) * 9 + i];
        	    			}
        	    			buffer0[sc * 2 + dc] = tmp + lde0_b[sc * 2 + dc];
        	    		}
        	    	}
        	    	for(int dc = 0; dc < hidden_c; ++dc)
        	    	{
        	    		float tmp = 0f;
        	    		for(int sc = 0; sc < hidden_c; ++sc)
        	    		{
        	    			tmp += buffer0[sc] * lde1_w[dc * hidden_c + sc];
        	    		}
        	    		tmp += lde1_b[dc];
        	    		// GELU
        				int sign;
        				float x_;
        				if(tmp < 0)
        				{
        					sign = -1;
        					x_ = -tmp / sqrt2;
        				}
        				else
        				{
        					sign = 1;
        					x_ = tmp / sqrt2;
        				}
        				float t = 1f / (1f + pc * x_);
        				float y_ = 1f - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (float)Math.Exp(-x_ * x_);
        				buffer1[dc] = tmp * (sign * y_ + 1f) * 0.5f;
        	    	}
        	    	for(int dc = 0; dc < src_c; ++dc)
        	    	{
        	    		float tmp = 0f;
        	    		for(int sc = 0; sc < hidden_c; ++sc)
        	    		{
        	    			tmp += buffer1[sc] * lde2_w[dc * hidden_c + sc];
        	    		}
        	    		buffer2[dc] = tmp + lde2_b[dc] + x[(dc * src_h + dy) * src_w + dx];
        	    	}
        	    	for(int dc = 0; dc < src_c; ++dc)
        	    	{
        	    		float tmp = 0f;
        	    		for(int sc = 0; sc < src_c; ++sc)
        	    		{
        	    			tmp += buffer2[sc] * l2_w[dc * src_c + sc];
        	    		}
        	    		dst[(dc * src_h + dy) * src_w + dx] = src[(dc * src_h + dy) * src_w + dx] + tmp + l2_b[dc];
        	    	}
        	    }
            });
        	Marshal.FreeCoTaskMem((IntPtr)y);
        	Marshal.FreeCoTaskMem((IntPtr)x);
        	GC.KeepAlive(input);
        	return output;
        }

        private static Tensor __residual_smfa_norm__(Tensor input,
				                                     Tensor linear_0_weight,
				                                     Tensor linear_0_bias,
					                                 Tensor linear_1_weight,
					                                 Tensor linear_1_bias,
					                                 Tensor linear_2_weight,
					                                 Tensor linear_2_bias,
					                                 Tensor lde_conv_0_0_weight,
					                                 Tensor lde_conv_0_0_bias,
					                                 Tensor lde_conv_0_1_weight,
					                                 Tensor lde_conv_0_1_bias,
					                                 Tensor lde_conv_1_weight,
					                                 Tensor lde_conv_1_bias,
					                                 Tensor dw_conv_weight,
					                                 Tensor dw_conv_bias,
					                                 Tensor alpha,
					                                 Tensor belt)
        {
        	int src_c = input.shape_ptr[0];
        	int src_h = input.shape_ptr[1];
        	int src_w = input.shape_ptr[2];
        	float* src = input.data_ptr;
        	// y, x = linear_0(input).chunk(2, dim=1)
        	float* l0_w = linear_0_weight.data_ptr;
        	float* l0_b = linear_0_bias.data_ptr;
        	float* y = (float*)Marshal.AllocCoTaskMem(src_c * src_h * src_w * sizeof(float));
        	float* x = (float*)Marshal.AllocCoTaskMem(src_c * src_h * src_w * sizeof(float));
        	Parallel.For(0, src_h, (int dy) =>
            {
        	    const float eps = 1e-12f;
        	    float* buffer = stackalloc float[src_c];
        		for(int dx = 0; dx < src_w; ++dx)
        		{
        			float tmp = 0f;
        			for(int sc = 0; sc < src_c; ++sc)
        			{
        				tmp += src[(sc * src_h + dy) * src_w + dx] * src[(sc * src_h + dy) * src_w + dx];
        			}
        			tmp = (float)Math.Sqrt(tmp);
        			if(tmp < eps)
        			{
        				tmp = eps;
        			}
        			for(int sc = 0; sc < src_c; ++sc)
        			{
        				buffer[sc] = src[(sc * src_h + dy) * src_w + dx] / tmp;
        			}
        			for(int dc = 0; dc < src_c; ++dc)
        			{
        				tmp = 0f;
        				for(int sc = 0; sc < src_c; ++sc)
        				{
        					tmp += buffer[sc] * l0_w[dc * src_c + sc];
        				}
        				y[(dc * src_h + dy) * src_w + dx] = tmp + l0_b[dc];
        			}
        			for(int dc = 0; dc < src_c; ++dc)
        			{
        				tmp = 0f;
        				for(int sc = 0; sc < src_c; ++sc)
        				{
        					tmp += buffer[sc] * l0_w[(src_c + dc) * src_c + sc];
        				}
        				x[(dc * src_h + dy) * src_w + dx] = tmp + l0_b[src_c + dc];
        			}
        		}
            });
        	// x_s = dw_conv(F.adaptive_max_pool2d(x, (h // down_scale, w // down_scale)))
        	// x_v = torch.var(x, dim=(-2,-1), keepdim=True)
        	// x_s = x_s * alpha + x_v * belt;
        	float* x_s = (float*)Marshal.AllocCoTaskMem(src_c * (src_h / 8) * (src_w / 8) * sizeof(float));
        	float* dw_w = dw_conv_weight.data_ptr;
        	float* dw_b = dw_conv_bias.data_ptr;
        	float* a = alpha.data_ptr;
        	float* b = belt.data_ptr;
        	Parallel.For(0, src_c, (int sc) =>
        	{
        	    float tmp = 0f;
        	    for(int sy = 0; sy < src_h; ++sy)
        	    {
        	    	for(int sx = 0; sx < src_w; ++sx)
        	    	{
        	    		tmp += x[(sc * src_h + sy) * src_w + sx];
        	    	}
        	    }
        	    tmp /= src_h * src_w;
        	    float x_v = 0f;
        	    for(int sy = 0; sy < src_h; ++sy)
        	    {
        	    	for(int sx = 0; sx < src_w; ++sx)
        	    	{
        	    		x_v += (x[(sc * src_h + sy) * src_w + sx] - tmp) * (x[(sc * src_h + sy) * src_w + sx] - tmp);
        	    	}
        	    }
        	    x_v /= src_h * src_w - 1;
        		float* buffer = stackalloc float[(src_h / 8) * (src_w / 8)];
        		for(int dy = 0; dy < src_h / 8; ++dy)
        		{
        			int sy0 = (dy / (src_h / 8)) * src_h + ((dy % (src_h / 8)) * src_h) / (src_h / 8);
        			int sy1 = 1 + ((dy + 1) * src_h - 1) / (src_h / 8);
        			for(int dx = 0; dx < src_w / 8; ++dx)
        			{
        				int sx0 = (dx / (src_w / 8)) * src_w + ((dx % (src_w / 8)) * src_w) / (src_w / 8);
        				int sx1 = 1 + ((dx + 1) * src_w - 1) / (src_w / 8);
        				float max = float.MinValue;
        				for(int sy = sy0; sy < sy1; ++sy)
        				{
        					for(int sx = sx0; sx < sx1; ++sx)
        					{
        						if(x[(sc * src_h + sy) * src_w + sx] > max)
        						{
        							max = x[(sc * src_h + sy) * src_w + sx];
        						}
        					}
        				}
        				buffer[dy * (src_w / 8) + dx] = max;
        			}
        		}
        		for(int dy = 0; dy < src_h / 8; ++dy)
        		{
        			for(int dx = 0; dx < src_w / 8; ++dx)
        			{
        				tmp = 0f;
        				for(int ky = 0; ky < 3; ++ky)
        				{
        					int sy = dy + ky - 1;
        					if((sy < 0) || (sy >= src_h / 8))
        					{
        						continue;
        					}
        					for(int kx = 0; kx < 3; ++kx)
        					{
        						int sx = dx + kx - 1;
	        					if((sx < 0) || (sx >= src_w / 8))
	        					{
	        						continue;
	        					}
	        					tmp += buffer[sy * (src_w / 8) + sx] * dw_w[(sc * 3 + ky) * 3 + kx];
        					}
        				}
        				x_s[(sc * (src_h / 8) + dy) * (src_w / 8) + dx] = (tmp + dw_b[sc]) * a[sc] + x_v * b[sc];
        			}
        		}
        	});
        	// x_s = GELU(linear_1(x_s))
        	float* l1_w = linear_1_weight.data_ptr;
        	float* l1_b = linear_1_bias.data_ptr;
        	Parallel.For(0, src_h / 8, (int dy) =>
        	{
        	    const float a1 = 0.254829592f;
        	    const float a2 = -0.284496736f;
        	    const float a3 = 1.421413741f;
        	    const float a4 = -1.453152027f;
        	    const float a5 = 1.061405429f;
        	    const float pc = 0.3275911f;
        	    const float sqrt2 = 1.41421356f;
        		float* buffer = stackalloc float[src_c];
        		for(int dx = 0; dx < src_w / 8; ++dx)
        		{
        			for(int sc = 0; sc < src_c; ++sc)
        			{
        				buffer[sc] = x_s[(sc * (src_h / 8) + dy) * (src_w / 8) + dx];
        			}
        			for(int dc = 0; dc < src_c; ++dc)
        			{
        				float tmp = 0f;
        				for(int sc = 0; sc < src_c; ++sc)
        				{
        					tmp += buffer[sc] * l1_w[dc * src_c + sc];
        				}
        				tmp += l1_b[dc];
        				// GELU
        				int sign;
        				float x_;
        				if(tmp < 0)
        				{
        					sign = -1;
        					x_ = -tmp / sqrt2;
        				}
        				else
        				{
        					sign = 1;
        					x_ = tmp / sqrt2;
        				}
        				float t = 1f / (1f + pc * x_);
        				float y_ = 1f - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (float)Math.Exp(-x_ * x_);
        				x_s[(dc * (src_h / 8) + dy) * (src_w / 8) + dx] = tmp * (sign * y_ + 1f) * 0.5f;
        			}
        		}
        	});
        	// x = x * F.interpolate(x_s, size=(h,w), mode='nearest')
        	Parallel.For(0, src_c, (int dc) =>
        	{
        		for(int dy = 0; dy < src_h; ++dy)
        		{
        			int sy = Math.Max(Math.Min((int)Math.Floor(dy * ((float)(src_h / 8) / src_h)), src_h - 1), 0);
        			for(int dx = 0; dx < src_w; ++dx)
        			{
        				int sx = Math.Max(Math.Min((int)Math.Floor(dx * ((float)(src_w / 8) / src_w)), src_w - 1), 0);
        				x[(dc * src_h + dy) * src_w + dx] *= x_s[(dc * (src_h / 8) + sy) * (src_w / 8) + sx];
        			}
        		}
        	});
        	Marshal.FreeCoTaskMem((IntPtr)x_s);
        	// y_d = lde(y)
        	// return linear_2(x + y_d) + input
        	int hidden_c = lde_conv_0_0_weight.shape_ptr[0];
        	float* lde0_w = lde_conv_0_0_weight.data_ptr;
        	float* lde0_b = lde_conv_0_0_bias.data_ptr;
        	float* lde1_w = lde_conv_0_1_weight.data_ptr;
        	float* lde1_b = lde_conv_0_1_bias.data_ptr;
        	float* lde2_w = lde_conv_1_weight.data_ptr;
        	float* lde2_b = lde_conv_1_bias.data_ptr;
        	float* l2_w = linear_2_weight.data_ptr;
        	float* l2_b = linear_2_bias.data_ptr;
        	Parallel.For(0, src_h, (int dy) =>
            {
        	    const float a1 = 0.254829592f;
        	    const float a2 = -0.284496736f;
        	    const float a3 = 1.421413741f;
        	    const float a4 = -1.453152027f;
        	    const float a5 = 1.061405429f;
        	    const float pc = 0.3275911f;
        	    const float sqrt2 = 1.41421356f;
        	    float* buffer0 = stackalloc float[hidden_c];
        	    float* buffer1 = stackalloc float[hidden_c];
        	    float* buffer2 = stackalloc float[src_c];
        	    float* buffer3 = stackalloc float[9];
        	    for(int dx = 0; dx < src_w; ++dx)
        	    {
        	    	for(int sc = 0; sc < src_c; ++sc)
        	    	{
        	    		for(int ky = 0; ky < 3; ++ky)
        	    		{
        	    			int sy = dy + ky - 1;
        	    			if((sy < 0) || (sy >= src_h))
        	    			{
        	    				*buffer3++ = 0f;
        	    				*buffer3++ = 0f;
        	    				*buffer3++ = 0f;
        	    				continue;
        	    			}
        	    			for(int kx = 0; kx < 3; ++kx)
        	    			{
        	    				int sx = dx + kx - 1;
	        	    			if((sx < 0) || (sx >= src_w))
	        	    			{
	        	    				*buffer3++ = 0f;
	        	    				continue;
	        	    			}
        	    				*buffer3++ = y[(sc * src_h + sy) * src_w + sx];
        	    			}
        	    		}
        	    		buffer3 -= 9;
        	    		for(int dc = 0; dc < hidden_c / src_c; ++dc)
        	    		{
        	    			float tmp = 0f;
        	    			for(int i = 0; i < 9; ++i)
        	    			{
        	    				tmp += buffer3[i] * lde0_w[(sc * 2 + dc) * 9 + i];
        	    			}
        	    			buffer0[sc * 2 + dc] = tmp + lde0_b[sc * 2 + dc];
        	    		}
        	    	}
        	    	for(int dc = 0; dc < hidden_c; ++dc)
        	    	{
        	    		float tmp = 0f;
        	    		for(int sc = 0; sc < hidden_c; ++sc)
        	    		{
        	    			tmp += buffer0[sc] * lde1_w[dc * hidden_c + sc];
        	    		}
        	    		tmp += lde1_b[dc];
        	    		// GELU
        				int sign;
        				float x_;
        				if(tmp < 0)
        				{
        					sign = -1;
        					x_ = -tmp / sqrt2;
        				}
        				else
        				{
        					sign = 1;
        					x_ = tmp / sqrt2;
        				}
        				float t = 1f / (1f + pc * x_);
        				float y_ = 1f - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (float)Math.Exp(-x_ * x_);
        				buffer1[dc] = tmp * (sign * y_ + 1f) * 0.5f;
        	    	}
        	    	for(int dc = 0; dc < src_c; ++dc)
        	    	{
        	    		float tmp = 0f;
        	    		for(int sc = 0; sc < hidden_c; ++sc)
        	    		{
        	    			tmp += buffer1[sc] * lde2_w[dc * hidden_c + sc];
        	    		}
        	    		buffer2[dc] = tmp + lde2_b[dc] + x[(dc * src_h + dy) * src_w + dx];
        	    	}
        	    	for(int dc = 0; dc < src_c; ++dc)
        	    	{
        	    		float tmp = 0f;
        	    		for(int sc = 0; sc < src_c; ++sc)
        	    		{
        	    			tmp += buffer2[sc] * l2_w[dc * src_c + sc];
        	    		}
        	    		src[(dc * src_h + dy) * src_w + dx] += tmp + l2_b[dc];
        	    	}
        	    }
            });
        	Marshal.FreeCoTaskMem((IntPtr)y);
        	Marshal.FreeCoTaskMem((IntPtr)x);
        	return input;
        }

        private static Tensor __residual_pcfn_norm__(Tensor input,
                                                     Tensor conv_0_weight,
                                                     Tensor conv_0_bias,
                                                     Tensor conv_1_weight,
                                                     Tensor conv_1_bias,
                                                     Tensor conv_2_weight,
                                                     Tensor conv_2_bias)
        {
        	int src_c = input.shape_ptr[0];
        	int src_h = input.shape_ptr[1];
        	int src_w = input.shape_ptr[2];
        	// Compute hidden = GELU(conv_0(F.normalize(x)))
        	int hidden_c = conv_0_weight.shape_ptr[0];
        	float* src = input.data_ptr;
        	float* w0 = conv_0_weight.data_ptr;
        	float* b0 = conv_0_bias.data_ptr;
        	float* hidden = (float*)Marshal.AllocCoTaskMem(hidden_c * src_h * src_w * sizeof(float));
        	Parallel.For(0, src_h, (int dy) =>
        	{
        	    const float eps = 1e-12f;
        	    const float a1 = 0.254829592f;
        	    const float a2 = -0.284496736f;
        	    const float a3 = 1.421413741f;
        	    const float a4 = -1.453152027f;
        	    const float a5 = 1.061405429f;
        	    const float pc = 0.3275911f;
        	    const float sqrt2 = 1.41421356f;
        	    float* buffer = stackalloc float[src_c];
        		for(int dx = 0; dx < src_w; ++dx)
        		{
        			float tmp = 0f;
        			for(int sc = 0; sc < src_c; ++sc)
        			{
        				tmp += src[(sc * src_h + dy) * src_w + dx] * src[(sc * src_h + dy) * src_w + dx];
        			}
        			tmp = (float)Math.Sqrt(tmp);
        			if(tmp < eps)
        			{
        				tmp = eps;
        			}
        			for(int c = 0; c < src_c; ++c)
        			{
        				buffer[c] = src[(c * src_h + dy) * src_w + dx] / tmp;
        			}
        			for(int dc = 0; dc < hidden_c; ++dc)
        			{
        				tmp = 0f;
        				for(int sc = 0; sc < src_c; ++sc)
        				{
        					tmp += buffer[sc] * w0[dc * src_c + sc];
        				}
        				tmp += b0[dc];
        				// GELU
        				int sign;
        				float x;
        				if(tmp < 0)
        				{
        					sign = -1;
        					x = -tmp / sqrt2;
        				}
        				else
        				{
        					sign = 1;
        					x = tmp / sqrt2;
        				}
        				float t = 1f / (1f + pc * x);
        				float y = 1f - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (float)Math.Exp(-x * x);
        				hidden[(dc * src_h + dy) * src_w + dx] = tmp * (sign * y + 1f) * 0.5f;
        			}
        		}
        	});
        	// Compute:
        	//          hidden[:self.p_dim, :, :] = GELU(conv_1(hidden[:p_dim, :, :]))
			//        	input = input + conv_2(hidden[:p_dim, :, :])
        	int p = conv_1_weight.shape_ptr[0];
        	float* w1 = conv_1_weight.data_ptr;
        	float* b1 = conv_1_bias.data_ptr;
        	float* w2 = conv_2_weight.data_ptr;
        	float* b2 = conv_2_bias.data_ptr;
        	Parallel.For(0, src_h, (int dy) =>
        	{
        	    const float a1 = 0.254829592f;
        	    const float a2 = -0.284496736f;
        	    const float a3 = 1.421413741f;
        	    const float a4 = -1.453152027f;
        	    const float a5 = 1.061405429f;
        	    const float pc = 0.3275911f;
        	    const float sqrt2 = 1.41421356f;
        	    float* buffer = stackalloc float[hidden_c];
        	    float* patch2vec = stackalloc float[p * 9];
        		for(int dx = 0; dx < src_w; ++dx)
        		{
        			// hidden[:self.p_dim, :, :] = GELU(conv_1(hidden[:p_dim, :, :]))
        			for(int sc = 0; sc < p; ++sc)
        			{
	        			for(int ky = 0; ky < 3; ++ky)
	    				{
	    					int sy = dy + ky - 1;
	    					if((sy < 0) || (sy >= src_h))
	    					{
	    						for(int kx = 0; kx < 3; ++kx)
	    						{
	    							*patch2vec++ = 0f;
	    						}
	    						continue;
	    					}
	    					for(int kx = 0; kx < 3; ++kx)
	    					{
	    						int sx = dx + kx - 1;
	    						if((sx < 0) || (sx >= src_w))
	        					{
	    							*patch2vec++ = 0f;
	        						continue;
	        					}
	    						*patch2vec++ = hidden[(sc * src_h + sy) * src_w + sx];
	    					}
	    				}
        			}
        			patch2vec -= p * 9;
        			for(int dc = 0; dc < p; ++dc)
        			{
        				float tmp = 0f;
        				for(int i = 0; i < p * 9; ++i)
        				{
        					tmp += patch2vec[i] * w1[dc * p * 9 + i];
        				}
        				tmp += b1[dc];
        				// GELU
        				int sign;
        				float x;
        				if(tmp < 0)
        				{
        					sign = -1;
        					x = -tmp / sqrt2;
        				}
        				else
        				{
        					sign = 1;
        					x = tmp / sqrt2;
        				}
        				float t = 1f / (1f + pc * x);
        				float y = 1f - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (float)Math.Exp(-x * x);
        				buffer[dc] = tmp * (sign * y + 1f) * 0.5f;
        			}
        			for(int dc = p; dc < hidden_c; ++dc)
        			{
        				buffer[dc] = hidden[(dc * src_h + dy) * src_w + dx];
        			}
        			// input = input + conv_2(hidden)
        			for(int dc = 0; dc < src_c; ++dc)
        			{
        				float tmp = 0f;
        				for(int sc = 0; sc < hidden_c; ++sc)
        				{
        					tmp += buffer[sc] * w2[dc * hidden_c + sc];
        				}
        				src[(dc * src_h + dy) * src_w + dx] += tmp + b2[dc];
        			}
        		}
        	});
        	Marshal.FreeCoTaskMem((IntPtr)hidden);
        	return input;
        }

        private static Tensor __fmb(Tensor input,
                                    Tensor smfa_linear_0_weight,
                                    Tensor smfa_linear_0_bias,
	                                Tensor smfa_linear_1_weight,
	                                Tensor smfa_linear_1_bias,
	                                Tensor smfa_linear_2_weight,
	                                Tensor smfa_linear_2_bias,
	                                Tensor smfa_lde_conv_0_0_weight,
	                                Tensor smfa_lde_conv_0_0_bias,
	                                Tensor smfa_lde_conv_0_1_weight,
	                                Tensor smfa_lde_conv_0_1_bias,
	                                Tensor smfa_lde_conv_1_weight,
	                                Tensor smfa_lde_conv_1_bias,
	                                Tensor smfa_dw_conv_weight,
	                                Tensor smfa_dw_conv_bias,
	                                Tensor smfa_alpha,
	                                Tensor smfa_belt,
	                                Tensor pcfn_conv_0_weight,
	                                Tensor pcfn_conv_0_bias,
	                                Tensor pcfn_conv_1_weight,
	                                Tensor pcfn_conv_1_bias,
	                                Tensor pcfn_conv_2_weight,
	                                Tensor pcfn_conv_2_bias)
        {
        	// input = self.smfa(F.normalize(input)) + input
        	input = __residual_smfa_norm(input,
        	                             smfa_linear_0_weight,
	                                     smfa_linear_0_bias,
		                                 smfa_linear_1_weight,
		                                 smfa_linear_1_bias,
		                                 smfa_linear_2_weight,
		                                 smfa_linear_2_bias,
		                                 smfa_lde_conv_0_0_weight,
		                                 smfa_lde_conv_0_0_bias,
		                                 smfa_lde_conv_0_1_weight,
		                                 smfa_lde_conv_0_1_bias,
		                                 smfa_lde_conv_1_weight,
		                                 smfa_lde_conv_1_bias,
		                                 smfa_dw_conv_weight,
		                                 smfa_dw_conv_bias,
		                                 smfa_alpha,
		                                 smfa_belt);
        	// input = pcfn(F.normalize(input)) + input
        	return __residual_pcfn_norm__(input,
        	                              pcfn_conv_0_weight,
        	                              pcfn_conv_0_bias,
        	                              pcfn_conv_1_weight,
        	                              pcfn_conv_1_bias,
        	                              pcfn_conv_2_weight,
        	                              pcfn_conv_2_bias);
        }

        private static Tensor __fmb__(Tensor input,
                                      Tensor smfa_linear_0_weight,
                                      Tensor smfa_linear_0_bias,
	                                  Tensor smfa_linear_1_weight,
	                                  Tensor smfa_linear_1_bias,
	                                  Tensor smfa_linear_2_weight,
	                                  Tensor smfa_linear_2_bias,
	                                  Tensor smfa_lde_conv_0_0_weight,
	                                  Tensor smfa_lde_conv_0_0_bias,
	                                  Tensor smfa_lde_conv_0_1_weight,
	                                  Tensor smfa_lde_conv_0_1_bias,
	                                  Tensor smfa_lde_conv_1_weight,
	                                  Tensor smfa_lde_conv_1_bias,
	                                  Tensor smfa_dw_conv_weight,
	                                  Tensor smfa_dw_conv_bias,
	                                  Tensor smfa_alpha,
	                                  Tensor smfa_belt,
	                                  Tensor pcfn_conv_0_weight,
	                                  Tensor pcfn_conv_0_bias,
	                                  Tensor pcfn_conv_1_weight,
	                                  Tensor pcfn_conv_1_bias,
	                                  Tensor pcfn_conv_2_weight,
	                                  Tensor pcfn_conv_2_bias)
        {
        	// input = self.smfa(F.normalize(input)) + input
        	input = __residual_smfa_norm__(input,
        	                               smfa_linear_0_weight,
	                                       smfa_linear_0_bias,
		                                   smfa_linear_1_weight,
		                                   smfa_linear_1_bias,
		                                   smfa_linear_2_weight,
		                                   smfa_linear_2_bias,
		                                   smfa_lde_conv_0_0_weight,
		                                   smfa_lde_conv_0_0_bias,
		                                   smfa_lde_conv_0_1_weight,
		                                   smfa_lde_conv_0_1_bias,
		                                   smfa_lde_conv_1_weight,
		                                   smfa_lde_conv_1_bias,
		                                   smfa_dw_conv_weight,
		                                   smfa_dw_conv_bias,
		                                   smfa_alpha,
		                                   smfa_belt);
        	// input = pcfn(F.normalize(input)) + input
        	return __residual_pcfn_norm__(input,
        	                              pcfn_conv_0_weight,
        	                              pcfn_conv_0_bias,
        	                              pcfn_conv_1_weight,
        	                              pcfn_conv_1_bias,
        	                              pcfn_conv_2_weight,
        	                              pcfn_conv_2_bias);
        }

        private static Tensor __add__(Tensor input, Tensor residual)
        {
        	int src_c = input.shape_ptr[0];
        	int src_h = input.shape_ptr[1];
        	int src_w = input.shape_ptr[2];
        	float* src = input.data_ptr;
        	float* res = residual.data_ptr;
        	Parallel.For(0, src_c, (int dc) =>
            {
            	for(int dy = 0; dy < src_h; ++dy)
            	{
            		for(int dx = 0; dx < src_w; ++dx)
            		{
            			src[(dc * src_h + dy) * src_w + dx] += res[(dc * src_h + dy) * src_w + dx];
            		}
            	}
            });
        	GC.KeepAlive(residual);
        	return input;
        }

        private static Tensor __pixshuffle_f4_conv2k_k3s1(Tensor input,
                                                          Tensor weight,
                                                          Tensor bias)
        {
        	int src_c = input.shape_ptr[0];
        	int src_h = input.shape_ptr[1];
        	int src_w = input.shape_ptr[2];
        	int dst_c = weight.shape_ptr[0];
        	int dst_c_s = dst_c / 16;
        	int dst_h_s = src_h * 4;
        	int dst_w_s = src_w * 4;
        	Tensor output = new Tensor(dst_c_s, dst_h_s, dst_w_s);
        	float* src = input.data_ptr;
        	float* dst = output.data_ptr;
        	float* w = weight.data_ptr;
        	float* b = bias.data_ptr;
        	Parallel.For(0, src_h, (int dy) =>
            {
        	    float* buffer = stackalloc float[src_c * 9];
            	for(int dx = 0; dx < src_w; ++dx)
            	{
            		for(int sc = 0; sc < src_c; ++sc)
        			{
        				for(int ky = 0; ky < 3; ++ky)
        				{
        					int sy = dy + ky - 1;
        					if((sy < 0) || (sy >= src_h))
        					{
        						for(int kx = 0; kx < 3; ++kx)
        						{
        							*buffer++ = 0f;
        						}
        						continue;
        					}
        					for(int kx = 0; kx < 3; ++kx)
        					{
        						int sx = dx + kx - 1;
            					if((sx < 0) || (sx >= src_w))
            					{
            						*buffer++ = 0f;
            						continue;
            					}
            					*buffer++ = src[(sc * src_h + sy) * src_w + sx];
        					}
        				}
        			}
            		buffer -= src_c * 9;
            		for(int dc = 0; dc < dst_c; ++dc)
            		{
            			float tmp = 0f;
            			for(int i = 0; i < src_c * 9; ++i)
            			{
            				tmp += buffer[i] * w[dc * src_c * 9 + i];
            			}
            			dst[(dc / 16 * dst_h_s + dy * 4 + dc % 16 / 4) * dst_w_s + dx * 4 + dc % 16 % 4] = tmp + b[dc];
            		}
            	}
            });
        	GC.KeepAlive(input);
        	return output;
        }

        public Bitmap Process(Bitmap src, Action<int> state)
        {
            state(0);
            Tensor x = __bitmap2tensor(src);
            state(1);
            x = __conv2d_k3s1(x, this.__to_feat_weight, this.__to_feat_bias);
            state(8); // 1 + 7.07
            Tensor x0 = __fmb(x,
                              this.__feats_0_smfa_linear_0_weight,
                              this.__feats_0_smfa_linear_0_bias,
                              this.__feats_0_smfa_linear_1_weight,
                              this.__feats_0_smfa_linear_1_bias,
                              this.__feats_0_smfa_linear_2_weight,
                              this.__feats_0_smfa_linear_2_bias,
                              this.__feats_0_smfa_lde_conv_0_0_weight,
                              this.__feats_0_smfa_lde_conv_0_0_bias,
                              this.__feats_0_smfa_lde_conv_0_1_weight,
                              this.__feats_0_smfa_lde_conv_0_1_bias,
                              this.__feats_0_smfa_lde_conv_1_weight,
                              this.__feats_0_smfa_lde_conv_1_bias,
                              this.__feats_0_smfa_dw_conv_weight,
                              this.__feats_0_smfa_dw_conv_bias,
                              this.__feats_0_smfa_alpha,
                              this.__feats_0_smfa_belt,
                              this.__feats_0_pcfn_conv_0_weight,
                              this.__feats_0_pcfn_conv_0_bias,
                              this.__feats_0_pcfn_conv_1_weight,
                              this.__feats_0_pcfn_conv_1_bias,
                              this.__feats_0_pcfn_conv_2_weight,
                              this.__feats_0_pcfn_conv_2_bias);
            state(15); // 8.07 + 7.07
            x0 = __fmb__(x0,
                         this.__feats_1_smfa_linear_0_weight,
                         this.__feats_1_smfa_linear_0_bias,
                         this.__feats_1_smfa_linear_1_weight,
                         this.__feats_1_smfa_linear_1_bias,
                         this.__feats_1_smfa_linear_2_weight,
                         this.__feats_1_smfa_linear_2_bias,
                         this.__feats_1_smfa_lde_conv_0_0_weight,
                         this.__feats_1_smfa_lde_conv_0_0_bias,
                         this.__feats_1_smfa_lde_conv_0_1_weight,
                         this.__feats_1_smfa_lde_conv_0_1_bias,
                         this.__feats_1_smfa_lde_conv_1_weight,
                         this.__feats_1_smfa_lde_conv_1_bias,
                         this.__feats_1_smfa_dw_conv_weight,
                         this.__feats_1_smfa_dw_conv_bias,
                         this.__feats_1_smfa_alpha,
                         this.__feats_1_smfa_belt,
                         this.__feats_1_pcfn_conv_0_weight,
                         this.__feats_1_pcfn_conv_0_bias,
                         this.__feats_1_pcfn_conv_1_weight,
                         this.__feats_1_pcfn_conv_1_bias,
                         this.__feats_1_pcfn_conv_2_weight,
                         this.__feats_1_pcfn_conv_2_bias);
            state(22); // 15.14 + 7.07
            x0 = __fmb__(x0,
                         this.__feats_2_smfa_linear_0_weight,
                         this.__feats_2_smfa_linear_0_bias,
                         this.__feats_2_smfa_linear_1_weight,
                         this.__feats_2_smfa_linear_1_bias,
                         this.__feats_2_smfa_linear_2_weight,
                         this.__feats_2_smfa_linear_2_bias,
                         this.__feats_2_smfa_lde_conv_0_0_weight,
                         this.__feats_2_smfa_lde_conv_0_0_bias,
                         this.__feats_2_smfa_lde_conv_0_1_weight,
                         this.__feats_2_smfa_lde_conv_0_1_bias,
                         this.__feats_2_smfa_lde_conv_1_weight,
                         this.__feats_2_smfa_lde_conv_1_bias,
                         this.__feats_2_smfa_dw_conv_weight,
                         this.__feats_2_smfa_dw_conv_bias,
                         this.__feats_2_smfa_alpha,
                         this.__feats_2_smfa_belt,
                         this.__feats_2_pcfn_conv_0_weight,
                         this.__feats_2_pcfn_conv_0_bias,
                         this.__feats_2_pcfn_conv_1_weight,
                         this.__feats_2_pcfn_conv_1_bias,
                         this.__feats_2_pcfn_conv_2_weight,
                         this.__feats_2_pcfn_conv_2_bias);
            state(29); // 22.21 + 7.07
            x0 = __fmb__(x0,
                         this.__feats_3_smfa_linear_0_weight,
                         this.__feats_3_smfa_linear_0_bias,
                         this.__feats_3_smfa_linear_1_weight,
                         this.__feats_3_smfa_linear_1_bias,
                         this.__feats_3_smfa_linear_2_weight,
                         this.__feats_3_smfa_linear_2_bias,
                         this.__feats_3_smfa_lde_conv_0_0_weight,
                         this.__feats_3_smfa_lde_conv_0_0_bias,
                         this.__feats_3_smfa_lde_conv_0_1_weight,
                         this.__feats_3_smfa_lde_conv_0_1_bias,
                         this.__feats_3_smfa_lde_conv_1_weight,
                         this.__feats_3_smfa_lde_conv_1_bias,
                         this.__feats_3_smfa_dw_conv_weight,
                         this.__feats_3_smfa_dw_conv_bias,
                         this.__feats_3_smfa_alpha,
                         this.__feats_3_smfa_belt,
                         this.__feats_3_pcfn_conv_0_weight,
                         this.__feats_3_pcfn_conv_0_bias,
                         this.__feats_3_pcfn_conv_1_weight,
                         this.__feats_3_pcfn_conv_1_bias,
                         this.__feats_3_pcfn_conv_2_weight,
                         this.__feats_3_pcfn_conv_2_bias);
            state(36); // 29.28 + 7.07
            x0 = __fmb__(x0,
                         this.__feats_4_smfa_linear_0_weight,
                         this.__feats_4_smfa_linear_0_bias,
                         this.__feats_4_smfa_linear_1_weight,
                         this.__feats_4_smfa_linear_1_bias,
                         this.__feats_4_smfa_linear_2_weight,
                         this.__feats_4_smfa_linear_2_bias,
                         this.__feats_4_smfa_lde_conv_0_0_weight,
                         this.__feats_4_smfa_lde_conv_0_0_bias,
                         this.__feats_4_smfa_lde_conv_0_1_weight,
                         this.__feats_4_smfa_lde_conv_0_1_bias,
                         this.__feats_4_smfa_lde_conv_1_weight,
                         this.__feats_4_smfa_lde_conv_1_bias,
                         this.__feats_4_smfa_dw_conv_weight,
                         this.__feats_4_smfa_dw_conv_bias,
                         this.__feats_4_smfa_alpha,
                         this.__feats_4_smfa_belt,
                         this.__feats_4_pcfn_conv_0_weight,
                         this.__feats_4_pcfn_conv_0_bias,
                         this.__feats_4_pcfn_conv_1_weight,
                         this.__feats_4_pcfn_conv_1_bias,
                         this.__feats_4_pcfn_conv_2_weight,
                         this.__feats_4_pcfn_conv_2_bias);
            state(43); // 36.35 + 7.07
            x0 = __fmb__(x0,
                         this.__feats_5_smfa_linear_0_weight,
                         this.__feats_5_smfa_linear_0_bias,
                         this.__feats_5_smfa_linear_1_weight,
                         this.__feats_5_smfa_linear_1_bias,
                         this.__feats_5_smfa_linear_2_weight,
                         this.__feats_5_smfa_linear_2_bias,
                         this.__feats_5_smfa_lde_conv_0_0_weight,
                         this.__feats_5_smfa_lde_conv_0_0_bias,
                         this.__feats_5_smfa_lde_conv_0_1_weight,
                         this.__feats_5_smfa_lde_conv_0_1_bias,
                         this.__feats_5_smfa_lde_conv_1_weight,
                         this.__feats_5_smfa_lde_conv_1_bias,
                         this.__feats_5_smfa_dw_conv_weight,
                         this.__feats_5_smfa_dw_conv_bias,
                         this.__feats_5_smfa_alpha,
                         this.__feats_5_smfa_belt,
                         this.__feats_5_pcfn_conv_0_weight,
                         this.__feats_5_pcfn_conv_0_bias,
                         this.__feats_5_pcfn_conv_1_weight,
                         this.__feats_5_pcfn_conv_1_bias,
                         this.__feats_5_pcfn_conv_2_weight,
                         this.__feats_5_pcfn_conv_2_bias);
            state(50); // 43.42 + 7.07
            x0 = __fmb__(x0,
                         this.__feats_6_smfa_linear_0_weight,
                         this.__feats_6_smfa_linear_0_bias,
                         this.__feats_6_smfa_linear_1_weight,
                         this.__feats_6_smfa_linear_1_bias,
                         this.__feats_6_smfa_linear_2_weight,
                         this.__feats_6_smfa_linear_2_bias,
                         this.__feats_6_smfa_lde_conv_0_0_weight,
                         this.__feats_6_smfa_lde_conv_0_0_bias,
                         this.__feats_6_smfa_lde_conv_0_1_weight,
                         this.__feats_6_smfa_lde_conv_0_1_bias,
                         this.__feats_6_smfa_lde_conv_1_weight,
                         this.__feats_6_smfa_lde_conv_1_bias,
                         this.__feats_6_smfa_dw_conv_weight,
                         this.__feats_6_smfa_dw_conv_bias,
                         this.__feats_6_smfa_alpha,
                         this.__feats_6_smfa_belt,
                         this.__feats_6_pcfn_conv_0_weight,
                         this.__feats_6_pcfn_conv_0_bias,
                         this.__feats_6_pcfn_conv_1_weight,
                         this.__feats_6_pcfn_conv_1_bias,
                         this.__feats_6_pcfn_conv_2_weight,
                         this.__feats_6_pcfn_conv_2_bias);
            state(58); // 50.49 + 7.07
            x0 = __fmb__(x0,
                         this.__feats_7_smfa_linear_0_weight,
                         this.__feats_7_smfa_linear_0_bias,
                         this.__feats_7_smfa_linear_1_weight,
                         this.__feats_7_smfa_linear_1_bias,
                         this.__feats_7_smfa_linear_2_weight,
                         this.__feats_7_smfa_linear_2_bias,
                         this.__feats_7_smfa_lde_conv_0_0_weight,
                         this.__feats_7_smfa_lde_conv_0_0_bias,
                         this.__feats_7_smfa_lde_conv_0_1_weight,
                         this.__feats_7_smfa_lde_conv_0_1_bias,
                         this.__feats_7_smfa_lde_conv_1_weight,
                         this.__feats_7_smfa_lde_conv_1_bias,
                         this.__feats_7_smfa_dw_conv_weight,
                         this.__feats_7_smfa_dw_conv_bias,
                         this.__feats_7_smfa_alpha,
                         this.__feats_7_smfa_belt,
                         this.__feats_7_pcfn_conv_0_weight,
                         this.__feats_7_pcfn_conv_0_bias,
                         this.__feats_7_pcfn_conv_1_weight,
                         this.__feats_7_pcfn_conv_1_bias,
                         this.__feats_7_pcfn_conv_2_weight,
                         this.__feats_7_pcfn_conv_2_bias);
            state(65); // 57.56 + 7.07
            x0 = __fmb__(x0,
                         this.__feats_8_smfa_linear_0_weight,
                         this.__feats_8_smfa_linear_0_bias,
                         this.__feats_8_smfa_linear_1_weight,
                         this.__feats_8_smfa_linear_1_bias,
                         this.__feats_8_smfa_linear_2_weight,
                         this.__feats_8_smfa_linear_2_bias,
                         this.__feats_8_smfa_lde_conv_0_0_weight,
                         this.__feats_8_smfa_lde_conv_0_0_bias,
                         this.__feats_8_smfa_lde_conv_0_1_weight,
                         this.__feats_8_smfa_lde_conv_0_1_bias,
                         this.__feats_8_smfa_lde_conv_1_weight,
                         this.__feats_8_smfa_lde_conv_1_bias,
                         this.__feats_8_smfa_dw_conv_weight,
                         this.__feats_8_smfa_dw_conv_bias,
                         this.__feats_8_smfa_alpha,
                         this.__feats_8_smfa_belt,
                         this.__feats_8_pcfn_conv_0_weight,
                         this.__feats_8_pcfn_conv_0_bias,
                         this.__feats_8_pcfn_conv_1_weight,
                         this.__feats_8_pcfn_conv_1_bias,
                         this.__feats_8_pcfn_conv_2_weight,
                         this.__feats_8_pcfn_conv_2_bias);
            state(72); // 64.63 + 7.07
            x0 = __fmb__(x0,
                         this.__feats_9_smfa_linear_0_weight,
                         this.__feats_9_smfa_linear_0_bias,
                         this.__feats_9_smfa_linear_1_weight,
                         this.__feats_9_smfa_linear_1_bias,
                         this.__feats_9_smfa_linear_2_weight,
                         this.__feats_9_smfa_linear_2_bias,
                         this.__feats_9_smfa_lde_conv_0_0_weight,
                         this.__feats_9_smfa_lde_conv_0_0_bias,
                         this.__feats_9_smfa_lde_conv_0_1_weight,
                         this.__feats_9_smfa_lde_conv_0_1_bias,
                         this.__feats_9_smfa_lde_conv_1_weight,
                         this.__feats_9_smfa_lde_conv_1_bias,
                         this.__feats_9_smfa_dw_conv_weight,
                         this.__feats_9_smfa_dw_conv_bias,
                         this.__feats_9_smfa_alpha,
                         this.__feats_9_smfa_belt,
                         this.__feats_9_pcfn_conv_0_weight,
                         this.__feats_9_pcfn_conv_0_bias,
                         this.__feats_9_pcfn_conv_1_weight,
                         this.__feats_9_pcfn_conv_1_bias,
                         this.__feats_9_pcfn_conv_2_weight,
                         this.__feats_9_pcfn_conv_2_bias);
            state(79); // 71.7 + 7.07
            x0 = __fmb__(x0,
                         this.__feats_10_smfa_linear_0_weight,
                         this.__feats_10_smfa_linear_0_bias,
                         this.__feats_10_smfa_linear_1_weight,
                         this.__feats_10_smfa_linear_1_bias,
                         this.__feats_10_smfa_linear_2_weight,
                         this.__feats_10_smfa_linear_2_bias,
                         this.__feats_10_smfa_lde_conv_0_0_weight,
                         this.__feats_10_smfa_lde_conv_0_0_bias,
                         this.__feats_10_smfa_lde_conv_0_1_weight,
                         this.__feats_10_smfa_lde_conv_0_1_bias,
                         this.__feats_10_smfa_lde_conv_1_weight,
                         this.__feats_10_smfa_lde_conv_1_bias,
                         this.__feats_10_smfa_dw_conv_weight,
                         this.__feats_10_smfa_dw_conv_bias,
                         this.__feats_10_smfa_alpha,
                         this.__feats_10_smfa_belt,
                         this.__feats_10_pcfn_conv_0_weight,
                         this.__feats_10_pcfn_conv_0_bias,
                         this.__feats_10_pcfn_conv_1_weight,
                         this.__feats_10_pcfn_conv_1_bias,
                         this.__feats_10_pcfn_conv_2_weight,
                         this.__feats_10_pcfn_conv_2_bias);
            state(86); // 78.77 + 7.07
            x0 = __fmb__(x0,
                         this.__feats_11_smfa_linear_0_weight,
                         this.__feats_11_smfa_linear_0_bias,
                         this.__feats_11_smfa_linear_1_weight,
                         this.__feats_11_smfa_linear_1_bias,
                         this.__feats_11_smfa_linear_2_weight,
                         this.__feats_11_smfa_linear_2_bias,
                         this.__feats_11_smfa_lde_conv_0_0_weight,
                         this.__feats_11_smfa_lde_conv_0_0_bias,
                         this.__feats_11_smfa_lde_conv_0_1_weight,
                         this.__feats_11_smfa_lde_conv_0_1_bias,
                         this.__feats_11_smfa_lde_conv_1_weight,
                         this.__feats_11_smfa_lde_conv_1_bias,
                         this.__feats_11_smfa_dw_conv_weight,
                         this.__feats_11_smfa_dw_conv_bias,
                         this.__feats_11_smfa_alpha,
                         this.__feats_11_smfa_belt,
                         this.__feats_11_pcfn_conv_0_weight,
                         this.__feats_11_pcfn_conv_0_bias,
                         this.__feats_11_pcfn_conv_1_weight,
                         this.__feats_11_pcfn_conv_1_bias,
                         this.__feats_11_pcfn_conv_2_weight,
                         this.__feats_11_pcfn_conv_2_bias);
            state(93); // 85.84 + 7.07
            x = __add__(x0, x);
            x = __pixshuffle_f4_conv2k_k3s1(x,
                                            this.__to_img_0_weight,
                                            this.__to_img_0_bias);
            state(99);
            Bitmap result = __tensor2bitmap(x);
            state(100);
            return result;
        }

        #region Singleton

        public static readonly SMFANet Instance;

        static SMFANet()
        {
            Instance = new SMFANet();
        }

        #endregion

    }

}