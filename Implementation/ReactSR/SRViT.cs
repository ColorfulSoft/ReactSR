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

    internal unsafe sealed class SRViT
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

        private readonly Tensor __head_weight;

        private readonly Tensor __head_bias;

        private readonly Tensor __body_layers_0_0_pos_embed_weight;

        private readonly Tensor __body_layers_0_0_pos_embed_bias;

        private readonly Tensor __body_layers_0_0_norm1_weight;

        private readonly Tensor __body_layers_0_0_norm1_bias;

        private readonly Tensor __body_layers_0_0_attn_qkv_weight;

        private readonly Tensor __body_layers_0_0_attn_proj_out_weight;

        private readonly Tensor __body_layers_0_0_attn_proj_out_bias;

        private readonly Tensor __body_layers_0_0_norm2_weight;

        private readonly Tensor __body_layers_0_0_norm2_bias;

        private readonly Tensor __body_layers_0_0_mlp_fc_0_weight;

        private readonly Tensor __body_layers_0_0_mlp_fc_0_bias;

        private readonly Tensor __body_layers_0_0_mlp_fc_2_weight;

        private readonly Tensor __body_layers_0_0_mlp_fc_2_bias;

        private readonly Tensor __body_layers_0_1_net_0_weight;

        private readonly Tensor __body_layers_0_1_net_0_bias;

        private readonly Tensor __body_layers_0_1_net_2_weight;

        private readonly Tensor __body_layers_0_1_net_2_bias;

        private readonly Tensor __body_layers_0_1_net_4_weight;

        private readonly Tensor __body_layers_0_1_net_4_bias;

        private readonly Tensor __body_layers_1_0_pos_embed_weight;

        private readonly Tensor __body_layers_1_0_pos_embed_bias;

        private readonly Tensor __body_layers_1_0_norm1_weight;

        private readonly Tensor __body_layers_1_0_norm1_bias;

        private readonly Tensor __body_layers_1_0_attn_qkv_weight;

        private readonly Tensor __body_layers_1_0_attn_proj_out_weight;

        private readonly Tensor __body_layers_1_0_attn_proj_out_bias;

        private readonly Tensor __body_layers_1_0_norm2_weight;

        private readonly Tensor __body_layers_1_0_norm2_bias;

        private readonly Tensor __body_layers_1_0_mlp_fc_0_weight;

        private readonly Tensor __body_layers_1_0_mlp_fc_0_bias;

        private readonly Tensor __body_layers_1_0_mlp_fc_2_weight;

        private readonly Tensor __body_layers_1_0_mlp_fc_2_bias;

        private readonly Tensor __body_layers_1_1_net_0_weight;

        private readonly Tensor __body_layers_1_1_net_0_bias;

        private readonly Tensor __body_layers_1_1_net_2_weight;

        private readonly Tensor __body_layers_1_1_net_2_bias;

        private readonly Tensor __body_layers_1_1_net_4_weight;

        private readonly Tensor __body_layers_1_1_net_4_bias;

        private readonly Tensor __body_layers_2_0_pos_embed_weight;

        private readonly Tensor __body_layers_2_0_pos_embed_bias;

        private readonly Tensor __body_layers_2_0_norm1_weight;

        private readonly Tensor __body_layers_2_0_norm1_bias;

        private readonly Tensor __body_layers_2_0_attn_qkv_weight;

        private readonly Tensor __body_layers_2_0_attn_proj_out_weight;

        private readonly Tensor __body_layers_2_0_attn_proj_out_bias;

        private readonly Tensor __body_layers_2_0_norm2_weight;

        private readonly Tensor __body_layers_2_0_norm2_bias;

        private readonly Tensor __body_layers_2_0_mlp_fc_0_weight;

        private readonly Tensor __body_layers_2_0_mlp_fc_0_bias;

        private readonly Tensor __body_layers_2_0_mlp_fc_2_weight;

        private readonly Tensor __body_layers_2_0_mlp_fc_2_bias;

        private readonly Tensor __body_layers_2_1_net_0_weight;

        private readonly Tensor __body_layers_2_1_net_0_bias;

        private readonly Tensor __body_layers_2_1_net_2_weight;

        private readonly Tensor __body_layers_2_1_net_2_bias;

        private readonly Tensor __body_layers_2_1_net_4_weight;

        private readonly Tensor __body_layers_2_1_net_4_bias;

        private readonly Tensor __body_layers_3_0_pos_embed_weight;

        private readonly Tensor __body_layers_3_0_pos_embed_bias;

        private readonly Tensor __body_layers_3_0_norm1_weight;

        private readonly Tensor __body_layers_3_0_norm1_bias;

        private readonly Tensor __body_layers_3_0_attn_qkv_weight;

        private readonly Tensor __body_layers_3_0_attn_proj_out_weight;

        private readonly Tensor __body_layers_3_0_attn_proj_out_bias;

        private readonly Tensor __body_layers_3_0_norm2_weight;

        private readonly Tensor __body_layers_3_0_norm2_bias;

        private readonly Tensor __body_layers_3_0_mlp_fc_0_weight;

        private readonly Tensor __body_layers_3_0_mlp_fc_0_bias;

        private readonly Tensor __body_layers_3_0_mlp_fc_2_weight;

        private readonly Tensor __body_layers_3_0_mlp_fc_2_bias;

        private readonly Tensor __body_layers_3_1_net_0_weight;

        private readonly Tensor __body_layers_3_1_net_0_bias;

        private readonly Tensor __body_layers_3_1_net_2_weight;

        private readonly Tensor __body_layers_3_1_net_2_bias;

        private readonly Tensor __body_layers_3_1_net_4_weight;

        private readonly Tensor __body_layers_3_1_net_4_bias;

        private readonly Tensor __body_layers_4_0_pos_embed_weight;

        private readonly Tensor __body_layers_4_0_pos_embed_bias;

        private readonly Tensor __body_layers_4_0_norm1_weight;

        private readonly Tensor __body_layers_4_0_norm1_bias;

        private readonly Tensor __body_layers_4_0_attn_qkv_weight;

        private readonly Tensor __body_layers_4_0_attn_proj_out_weight;

        private readonly Tensor __body_layers_4_0_attn_proj_out_bias;

        private readonly Tensor __body_layers_4_0_norm2_weight;

        private readonly Tensor __body_layers_4_0_norm2_bias;

        private readonly Tensor __body_layers_4_0_mlp_fc_0_weight;

        private readonly Tensor __body_layers_4_0_mlp_fc_0_bias;

        private readonly Tensor __body_layers_4_0_mlp_fc_2_weight;

        private readonly Tensor __body_layers_4_0_mlp_fc_2_bias;

        private readonly Tensor __body_layers_4_1_net_0_weight;

        private readonly Tensor __body_layers_4_1_net_0_bias;

        private readonly Tensor __body_layers_4_1_net_2_weight;

        private readonly Tensor __body_layers_4_1_net_2_bias;

        private readonly Tensor __body_layers_4_1_net_4_weight;

        private readonly Tensor __body_layers_4_1_net_4_bias;

        private readonly Tensor __fuse_weight;

        private readonly Tensor __fuse_bias;

        private readonly Tensor __upsapling_0_weight;

        private readonly Tensor __upsapling_0_bias;

        private readonly Tensor __upsapling_2_weight;

        private readonly Tensor __upsapling_2_bias;

        private readonly Tensor __tail_weight;

        private readonly Tensor __tail_bias;

        private SRViT()
        {
            BinaryReader reader = new BinaryReader(Assembly.GetExecutingAssembly().GetManifestResourceStream("SRViT.hmodel"));
            __load_parameter(reader, ref this.__head_weight, 40, 3, 3, 3);
            __load_parameter(reader, ref this.__head_bias, 40);
            __load_parameter(reader, ref this.__body_layers_0_0_pos_embed_weight, 40, 1, 3, 3);
            __load_parameter(reader, ref this.__body_layers_0_0_pos_embed_bias, 40);
            __load_parameter(reader, ref this.__body_layers_0_0_norm1_weight, 40);
            __load_parameter(reader, ref this.__body_layers_0_0_norm1_bias, 40);
            __load_parameter(reader, ref this.__body_layers_0_0_attn_qkv_weight, 120, 40);
            __load_parameter(reader, ref this.__body_layers_0_0_attn_proj_out_weight, 40, 40);
            __load_parameter(reader, ref this.__body_layers_0_0_attn_proj_out_bias, 40);
            __load_parameter(reader, ref this.__body_layers_0_0_norm2_weight, 40);
            __load_parameter(reader, ref this.__body_layers_0_0_norm2_bias, 40);
            __load_parameter(reader, ref this.__body_layers_0_0_mlp_fc_0_weight, 160, 40);
            __load_parameter(reader, ref this.__body_layers_0_0_mlp_fc_0_bias, 160);
            __load_parameter(reader, ref this.__body_layers_0_0_mlp_fc_2_weight, 40, 160);
            __load_parameter(reader, ref this.__body_layers_0_0_mlp_fc_2_bias, 40);
            __load_parameter(reader, ref this.__body_layers_0_1_net_0_weight, 160, 40, 1, 1);
            __load_parameter(reader, ref this.__body_layers_0_1_net_0_bias, 160);
            __load_parameter(reader, ref this.__body_layers_0_1_net_2_weight, 160, 1, 3, 3);
            __load_parameter(reader, ref this.__body_layers_0_1_net_2_bias, 160);
            __load_parameter(reader, ref this.__body_layers_0_1_net_4_weight, 40, 160, 1, 1);
            __load_parameter(reader, ref this.__body_layers_0_1_net_4_bias, 40);
            __load_parameter(reader, ref this.__body_layers_1_0_pos_embed_weight, 40, 1, 3, 3);
            __load_parameter(reader, ref this.__body_layers_1_0_pos_embed_bias, 40);
            __load_parameter(reader, ref this.__body_layers_1_0_norm1_weight, 40);
            __load_parameter(reader, ref this.__body_layers_1_0_norm1_bias, 40);
            __load_parameter(reader, ref this.__body_layers_1_0_attn_qkv_weight, 120, 40);
            __load_parameter(reader, ref this.__body_layers_1_0_attn_proj_out_weight, 40, 40);
            __load_parameter(reader, ref this.__body_layers_1_0_attn_proj_out_bias, 40);
            __load_parameter(reader, ref this.__body_layers_1_0_norm2_weight, 40);
            __load_parameter(reader, ref this.__body_layers_1_0_norm2_bias, 40);
            __load_parameter(reader, ref this.__body_layers_1_0_mlp_fc_0_weight, 80, 40);
            __load_parameter(reader, ref this.__body_layers_1_0_mlp_fc_0_bias, 80);
            __load_parameter(reader, ref this.__body_layers_1_0_mlp_fc_2_weight, 40, 80);
            __load_parameter(reader, ref this.__body_layers_1_0_mlp_fc_2_bias, 40);
            __load_parameter(reader, ref this.__body_layers_1_1_net_0_weight, 80, 40, 1, 1);
            __load_parameter(reader, ref this.__body_layers_1_1_net_0_bias, 80);
            __load_parameter(reader, ref this.__body_layers_1_1_net_2_weight, 80, 1, 3, 3);
            __load_parameter(reader, ref this.__body_layers_1_1_net_2_bias, 80);
            __load_parameter(reader, ref this.__body_layers_1_1_net_4_weight, 40, 80, 1, 1);
            __load_parameter(reader, ref this.__body_layers_1_1_net_4_bias, 40);
            __load_parameter(reader, ref this.__body_layers_2_0_pos_embed_weight, 40, 1, 3, 3);
            __load_parameter(reader, ref this.__body_layers_2_0_pos_embed_bias, 40);
            __load_parameter(reader, ref this.__body_layers_2_0_norm1_weight, 40);
            __load_parameter(reader, ref this.__body_layers_2_0_norm1_bias, 40);
            __load_parameter(reader, ref this.__body_layers_2_0_attn_qkv_weight, 120, 40);
            __load_parameter(reader, ref this.__body_layers_2_0_attn_proj_out_weight, 40, 40);
            __load_parameter(reader, ref this.__body_layers_2_0_attn_proj_out_bias, 40);
            __load_parameter(reader, ref this.__body_layers_2_0_norm2_weight, 40);
            __load_parameter(reader, ref this.__body_layers_2_0_norm2_bias, 40);
            __load_parameter(reader, ref this.__body_layers_2_0_mlp_fc_0_weight, 80, 40);
            __load_parameter(reader, ref this.__body_layers_2_0_mlp_fc_0_bias, 80);
            __load_parameter(reader, ref this.__body_layers_2_0_mlp_fc_2_weight, 40, 80);
            __load_parameter(reader, ref this.__body_layers_2_0_mlp_fc_2_bias, 40);
            __load_parameter(reader, ref this.__body_layers_2_1_net_0_weight, 80, 40, 1, 1);
            __load_parameter(reader, ref this.__body_layers_2_1_net_0_bias, 80);
            __load_parameter(reader, ref this.__body_layers_2_1_net_2_weight, 80, 1, 3, 3);
            __load_parameter(reader, ref this.__body_layers_2_1_net_2_bias, 80);
            __load_parameter(reader, ref this.__body_layers_2_1_net_4_weight, 40, 80, 1, 1);
            __load_parameter(reader, ref this.__body_layers_2_1_net_4_bias, 40);
            __load_parameter(reader, ref this.__body_layers_3_0_pos_embed_weight, 40, 1, 3, 3);
            __load_parameter(reader, ref this.__body_layers_3_0_pos_embed_bias, 40);
            __load_parameter(reader, ref this.__body_layers_3_0_norm1_weight, 40);
            __load_parameter(reader, ref this.__body_layers_3_0_norm1_bias, 40);
            __load_parameter(reader, ref this.__body_layers_3_0_attn_qkv_weight, 120, 40);
            __load_parameter(reader, ref this.__body_layers_3_0_attn_proj_out_weight, 40, 40);
            __load_parameter(reader, ref this.__body_layers_3_0_attn_proj_out_bias, 40);
            __load_parameter(reader, ref this.__body_layers_3_0_norm2_weight, 40);
            __load_parameter(reader, ref this.__body_layers_3_0_norm2_bias, 40);
            __load_parameter(reader, ref this.__body_layers_3_0_mlp_fc_0_weight, 80, 40);
            __load_parameter(reader, ref this.__body_layers_3_0_mlp_fc_0_bias, 80);
            __load_parameter(reader, ref this.__body_layers_3_0_mlp_fc_2_weight, 40, 80);
            __load_parameter(reader, ref this.__body_layers_3_0_mlp_fc_2_bias, 40);
            __load_parameter(reader, ref this.__body_layers_3_1_net_0_weight, 80, 40, 1, 1);
            __load_parameter(reader, ref this.__body_layers_3_1_net_0_bias, 80);
            __load_parameter(reader, ref this.__body_layers_3_1_net_2_weight, 80, 1, 3, 3);
            __load_parameter(reader, ref this.__body_layers_3_1_net_2_bias, 80);
            __load_parameter(reader, ref this.__body_layers_3_1_net_4_weight, 40, 80, 1, 1);
            __load_parameter(reader, ref this.__body_layers_3_1_net_4_bias, 40);
            __load_parameter(reader, ref this.__body_layers_4_0_pos_embed_weight, 40, 1, 3, 3);
            __load_parameter(reader, ref this.__body_layers_4_0_pos_embed_bias, 40);
            __load_parameter(reader, ref this.__body_layers_4_0_norm1_weight, 40);
            __load_parameter(reader, ref this.__body_layers_4_0_norm1_bias, 40);
            __load_parameter(reader, ref this.__body_layers_4_0_attn_qkv_weight, 120, 40);
            __load_parameter(reader, ref this.__body_layers_4_0_attn_proj_out_weight, 40, 40);
            __load_parameter(reader, ref this.__body_layers_4_0_attn_proj_out_bias, 40);
            __load_parameter(reader, ref this.__body_layers_4_0_norm2_weight, 40);
            __load_parameter(reader, ref this.__body_layers_4_0_norm2_bias, 40);
            __load_parameter(reader, ref this.__body_layers_4_0_mlp_fc_0_weight, 160, 40);
            __load_parameter(reader, ref this.__body_layers_4_0_mlp_fc_0_bias, 160);
            __load_parameter(reader, ref this.__body_layers_4_0_mlp_fc_2_weight, 40, 160);
            __load_parameter(reader, ref this.__body_layers_4_0_mlp_fc_2_bias, 40);
            __load_parameter(reader, ref this.__body_layers_4_1_net_0_weight, 160, 40, 1, 1);
            __load_parameter(reader, ref this.__body_layers_4_1_net_0_bias, 160);
            __load_parameter(reader, ref this.__body_layers_4_1_net_2_weight, 160, 1, 3, 3);
            __load_parameter(reader, ref this.__body_layers_4_1_net_2_bias, 160);
            __load_parameter(reader, ref this.__body_layers_4_1_net_4_weight, 40, 160, 1, 1);
            __load_parameter(reader, ref this.__body_layers_4_1_net_4_bias, 40);
            __load_parameter(reader, ref this.__fuse_weight, 40, 80, 3, 3);
            __load_parameter(reader, ref this.__fuse_bias, 40);
            __load_parameter(reader, ref this.__upsapling_0_weight, 160, 40, 1, 1);
            __load_parameter(reader, ref this.__upsapling_0_bias, 160);
            __load_parameter(reader, ref this.__upsapling_2_weight, 160, 40, 1, 1);
            __load_parameter(reader, ref this.__upsapling_2_bias, 160);
            __load_parameter(reader, ref this.__tail_weight, 3, 40, 3, 3);
            __load_parameter(reader, ref this.__tail_bias, 3);
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

        private static Tensor __depthwise_conv2d_k3s1_(Tensor input, Tensor weight, Tensor bias)
        {
            int channels = input.shape_ptr[0];
            int height = input.shape_ptr[1];
            int width = input.shape_ptr[2];
            Tensor output = new Tensor(channels, height, width);
            float* src = input.data_ptr;
            float* w = weight.data_ptr;
            float* b = bias.data_ptr;
            float* dst = output.data_ptr;
            Parallel.For(0, channels, (int c) =>
            {
                float* src1 = src + c * height * width;
                float* w1 = w + c * 9;
                float* dst1 = dst + c * height * width;
                for(int dy = 0; dy < height; ++dy)
                {
                    float* src2_1 = src1 + dy * width;
                    float* dst2 = dst1 + dy * width;
                    for(int dx = 0; dx < width; ++dx)
                    {
                        float sum = 0;
                        for(int ky = 0; ky < 3; ky++)
                        {
                            int sy = dy + ky - 1;
                            if((sy < 0) || (sy >= height))
                            {
                                continue;
                            }
                            float* src2 = src1 + sy * width;
                            float* w2 = w1 + ky * 3;
                            for(int kx = 0; kx < 3; kx++)
                            {
                                int sx = dx + kx - 1;
                                if((sx >= 0) && (sx < width))
                                {
                                    sum += src2[sx] * w2[kx];
                                }
                            }
                        }
                        dst2[dx] = sum + b[c] + src2_1[dx];
                    }
                }
            });
            GC.KeepAlive(input);
            return output;
        }

        public static Tensor w_msa_norm(Tensor input,
                                        Tensor norm_weight,
                                        Tensor norm_bias,
                                        Tensor qkv_weight,
                                        Tensor proj_out_weight,
                                        Tensor proj_out_bias)
        {
            const int window_size = 8;
            const int n_heads = 8;
            int channels = input.shape_ptr[0];
            int src_h = input.shape_ptr[1];
            int src_w = input.shape_ptr[2];
            int pad_h = (window_size - src_h % window_size) % window_size;
            int pad_w = (window_size - src_w % window_size) % window_size;
            int head_channels = channels / n_heads;
            int h_blocks = (src_h + pad_h) / window_size;
            int w_blocks = (src_w + pad_w) / window_size;
            float* src = input.data_ptr;
            float* norm_w = norm_weight.data_ptr;
            float* norm_b = norm_bias.data_ptr;
            float* qkv = qkv_weight.data_ptr;
            float* proj_w = proj_out_weight.data_ptr;
            float* proj_b = proj_out_bias.data_ptr;
            float scale = (float)Math.Pow(head_channels, -0.5);
            Parallel.For(0, h_blocks, (int by) =>
            {
                float* patch = stackalloc float[channels * window_size * window_size];
                float* q = stackalloc float[channels * window_size * window_size];
                float* k = stackalloc float[channels * window_size * window_size];
                float* v = stackalloc float[channels * window_size * window_size];
                float* buf = stackalloc float[Math.Max(channels, window_size * window_size)];
                for(int bx = 0; bx < w_blocks; ++bx)
                {
                    // Get patch for window attention
                    for(int sc = 0; sc < channels; ++sc)
                    {
                        for(int dy = 0; dy < window_size; ++dy)
                        {
                            int sy = by * window_size + dy;
                            for(int dx = 0; dx < window_size; ++dx)
                            {
                                int sx = bx * window_size + dx;
                                if((sy >= src_h) || (sx >= src_w))
                                {
                                    patch[(sc * window_size + dy) * window_size + dx] = 0f;
                                    continue;
                                }
                                patch[(sc * window_size + dy) * window_size + dx] = src[(sc * src_h + sy) * src_w + sx];
                            }
                        }
                    }
                    // Compute weight norm and q, k, v
                    for(int n = 0; n < window_size * window_size; ++n)
                    {
                        float mean = 0f;
                        for(int sc = 0; sc < channels; ++sc)
                        {
                            buf[sc] = patch[sc * window_size * window_size + n];
                            mean += buf[sc];
                        }
                        mean /= channels;
                        // Get std
                        float std = 0f;
                        for(int sc = 0; sc < channels; ++sc)
                        {
                            float val = buf[sc] - mean;
                            std += val * val;
                        }
                        std = (float)Math.Sqrt(std / channels + 1e-5f);
                        // layer norm
                        for(int sc = 0; sc < channels; ++sc)
                        {
                            buf[sc] = (buf[sc] - mean) / std * norm_w[sc] + norm_b[sc];
                        }
                        // q, k, v
                        for(int dc = 0; dc < channels; ++dc)
                        {
                            float sum_q = 0f;
                            float sum_k = 0f;
                            float sum_v = 0f;
                            for(int sc = 0; sc < channels; ++sc)
                            {
                                sum_q += buf[sc] * qkv[dc * channels + sc];
                                sum_k += buf[sc] * qkv[(channels + dc) * channels + sc];
                                sum_v += buf[sc] * qkv[(channels * 2 + dc) * channels + sc];
                            }
                            q[dc * window_size * window_size + n] = sum_q;
                            k[dc * window_size * window_size + n] = sum_k;
                            v[dc * window_size * window_size + n] = sum_v;
                        }
                    }
                    // Compute attention
                    for(int h = 0; h < n_heads; ++h)
                    {
                        for(int nq = 0; nq < window_size * window_size; ++nq)
                        {
                            // q^T*k*scale
                            float sum = 0;
                            for(int nk = 0; nk < window_size * window_size; ++nk)
                            {
                                float act = 0f;
                                for(int c = h * head_channels; c < (h + 1) * head_channels; ++c)
                                {
                                    act += q[c * window_size * window_size + nq] * k[c * window_size * window_size + nk];
                                }
                                buf[nk] = (float)Math.Exp(act * scale);
                                sum += buf[nk];
                            }
                            // softmax
                            for(int nk = 0; nk < window_size * window_size; ++nk)
                            {
                                buf[nk] /= sum;
                            }
                            // out
                            for(int c = h * head_channels; c < (h + 1) * head_channels; ++c)
                            {
                                sum = 0f;
                                for(int nv = 0; nv < window_size * window_size; ++nv)
                                {
                                    sum += buf[nv] * v[c * window_size * window_size + nv];
                                }
                                patch[c * window_size * window_size + nq] = sum;
                            }
                        }
                    }
                    // project
                    for(int n = 0; n < window_size * window_size; ++n)
                    {
                        for(int sc = 0; sc < channels; ++sc)
                        {
                            buf[sc] = patch[sc * window_size * window_size + n];
                        }
                        for(int dc = 0; dc < channels; ++dc)
                        {
                            float sum = 0f;
                            for(int sc = 0; sc < channels; ++sc)
                            {
                                sum += buf[sc] * proj_w[dc * channels + sc];
                            }
                            patch[dc * window_size * window_size + n] = sum + proj_b[dc];
                        }
                    }
                    // set patch
                    for(int sc = 0; sc < channels; ++sc)
                    {
                        for(int dy = 0; dy < window_size; ++dy)
                        {
                            int sy = by * window_size + dy;
                            for(int dx = 0; dx < window_size; ++dx)
                            {
                                int sx = bx * window_size + dx;
                                if((sy >= src_h) || (sx >= src_w))
                                {
                                    continue;
                                }
                                src[(sc * src_h + sy) * src_w + sx] += patch[(sc * window_size + dy) * window_size + dx];
                            }
                        }
                    }
                }
            });
            GC.KeepAlive(input);
            return input;
        }

        public static Tensor mlp_norm_(Tensor input,
                                       Tensor norm_weight,
                                       Tensor norm_bias,
                                       Tensor fc1_weight,
                                       Tensor fc1_bias,
                                       Tensor fc2_weight,
                                       Tensor fc2_bias)
        {
            int height = input.shape_ptr[1];
            int width = input.shape_ptr[2];
            int channel = input.shape_ptr[0];
            int hidden_c = fc1_weight.shape_ptr[0];
            float* src = input.data_ptr;
            float* norm_w = norm_weight.data_ptr;
            float* norm_b = norm_bias.data_ptr;
            float* w1 = fc1_weight.data_ptr;
            float* b1 = fc1_bias.data_ptr;
            float* w2 = fc2_weight.data_ptr;
            float* b2 = fc2_bias.data_ptr;
            Parallel.For(0, height, (int y) =>
            {
                float* buf1 = stackalloc float[channel];
                float* buf2 = stackalloc float[hidden_c];
                for(int x = 0; x < width; ++x)
                {
                    // Get src channels vector and mean
                    float mean = 0f;
                    for(int c = 0; c < channel; ++c)
                    {
                        buf1[c] = src[(c * height + y) * width + x];
                        mean += buf1[c];
                    }
                    mean /= channel;
                    // Get std
                    float std = 0f;
                    for(int c = 0; c < channel; ++c)
                    {
                        float val = buf1[c] - mean;
                        std += val * val;
                    }
                    std = (float)Math.Sqrt(std / channel + 1e-5f);
                    // layer norm
                    for(int c = 0; c < channel; ++c)
                    {
                        buf1[c] = (buf1[c] - mean) / std * norm_w[c] + norm_b[c];
                    }
                    // gelu(fc1)
                    for(int dc = 0; dc < hidden_c; ++dc)
                    {
                        // fc1
                        float sum = 0f;
                        for(int c = 0; c < channel; ++c)
                        {
                            sum += buf1[c] * w1[dc * channel + c];
                        }
                        sum += b1[dc];
                        // gelu
                        const float a1 = 0.254829592f;
                        const float a2 = -0.284496736f;
                        const float a3 = 1.421413741f;
                        const float a4 = -1.453152027f;
                        const float a5 = 1.061405429f;
                        const float p = 0.3275911f;
                        float phi = sum / 1.414213562373095f;
                        int sign = 1;
                        if(phi < 0)
                        {
                            sign = -1;
                        }
                        phi = Math.Abs(phi);
                        float t = 1f / (1f + p * phi);
                        phi = 1f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (float)Math.Exp(-phi * phi);
                        sum *= (sign * phi + 1f) / 2f;
                        buf2[dc] = sum;
                    }
                    // fc2
                    for(int dc = 0; dc < channel; ++dc)
                    {
                        float sum = 0f;
                        for(int c = 0; c < hidden_c; ++c)
                        {
                            sum += buf2[c] * w2[dc * hidden_c + c];
                        }
                        src[(dc * height + y) * width + x] += sum + b2[dc];
                    }
                }
            });
            return input;
        }

        private static Tensor __transformer(Tensor input,
                                            Tensor pos_embed_weight,
                                            Tensor pos_embed_bias,
                                            Tensor norm1_weight,
                                            Tensor norm1_bias,
                                            Tensor qkv_weight,
                                            Tensor proj_out_weight,
                                            Tensor proj_out_bias,
                                            Tensor norm2_weight,
                                            Tensor norm2_bias,
                                            Tensor fc1_weight,
                                            Tensor fc1_bias,
                                            Tensor fc2_weight,
                                            Tensor fc2_bias)
        {
            Tensor x = input;
            x = __depthwise_conv2d_k3s1_(x, pos_embed_weight, pos_embed_bias);
            x = w_msa_norm(x, norm1_weight, norm1_bias, qkv_weight, proj_out_weight, proj_out_bias);
            x = mlp_norm_(x, norm2_weight, norm2_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias);
            return x;
        }

        private static Tensor __conv2d_k1s1_lrelu(Tensor input, Tensor weight, Tensor bias)
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
                float* buffer = stackalloc float[src_c];
                float* dst1 = dst + dy * width;
                for(int dx = 0; dx < width; ++dx)
                {
                    float* dst2 = dst1 + dx;
                    for(int sc = 0; sc < src_c; ++sc)
                    {
                        *buffer++ = src[(sc * height + dy) * width + dx];
                    }
                    buffer -= src_c;
                    for(int dc = 0; dc < dst_c; ++dc)
                    {
                        float* w1 = w + dc * src_c;
                        float sum = 0;
                        for(int sc = 0; sc < src_c; ++sc)
                        {
                            sum += buffer[sc] * w1[sc];
                        }
                        sum += b[dc];
                        dst2[dc * height * width] = (sum < 0f) ? sum * 0.2f : sum;
                    }
                }
            });
            GC.KeepAlive(input);
            return output;
        }

        private static Tensor __depthwise_conv2d_k3s1_lrelu_conv2d_k1s1_(Tensor residual, Tensor input, Tensor weight1, Tensor bias1, Tensor weight2, Tensor bias2)
        {
            int src_c = input.shape_ptr[0];
            int dst_c = residual.shape_ptr[0];
            int height = input.shape_ptr[1];
            int width = input.shape_ptr[2];
            float* res = residual.data_ptr;
            float* src = input.data_ptr;
            float* w1 = weight1.data_ptr;
            float* b1 = bias1.data_ptr;
            float* w2 = weight2.data_ptr;
            float* b2 = bias2.data_ptr;
            Parallel.For(0, height, (int dy) =>
            {
                float* buffer = stackalloc float[src_c];
                for(int dx = 0; dx < width; ++dx)
                {
                    for(int sc = 0; sc < src_c; ++sc)
                    {
                        float sum = 0;
                        for(int ky = 0; ky < 3; ky++)
                        {
                            int sy = dy + ky - 1;
                            if((sy < 0) || (sy >= height))
                            {
                                continue;
                            }
                            for(int kx = 0; kx < 3; kx++)
                            {
                                int sx = dx + kx - 1;
                                if((sx >= 0) && (sx < width))
                                {
                                    sum += src[(sc * height + sy) * width + sx] * w1[(sc * 3 + ky) * 3 + kx];
                                }
                            }
                        }
                        float val = sum + b1[sc];
                        buffer[sc] = (val < 0f) ? val * 0.2f : val;
                    }
                    for(int dc = 0; dc < dst_c; ++dc)
                    {
                        float sum = 0f;
                        for(int sc = 0; sc < src_c; ++sc)
                        {
                            sum += buffer[sc] * w2[dc * src_c + sc];
                        }
                        res[(dc * height + dy) * width + dx] += sum + b2[dc];
                    }
                }
            });
            GC.KeepAlive(input);
            return residual;
        }

        private static Tensor __residual_block(Tensor input,
                                               Tensor conv1_weight,
                                               Tensor conv1_bias,
                                               Tensor conv2_weight,
                                               Tensor conv2_bias,
                                               Tensor conv3_weight,
                                               Tensor conv3_bias)
        {
            Tensor residual = input;
            input = __conv2d_k1s1_lrelu(input, conv1_weight, conv1_bias);
            input = __depthwise_conv2d_k3s1_lrelu_conv2d_k1s1_(residual, input, conv2_weight, conv2_bias, conv3_weight, conv3_bias);
            return input;
        }

        private static Tensor __base_block(Tensor input,
                                           Tensor pos_embed_weight,
                                           Tensor pos_embed_bias,
                                           Tensor norm1_weight,
                                           Tensor norm1_bias,
                                           Tensor qkv_weight,
                                           Tensor proj_out_weight,
                                           Tensor proj_out_bias,
                                           Tensor norm2_weight,
                                           Tensor norm2_bias,
                                           Tensor fc1_weight,
                                           Tensor fc1_bias,
                                           Tensor fc2_weight,
                                           Tensor fc2_bias,
                                           Tensor residual_conv1_weight,
                                           Tensor residual_conv1_bias,
                                           Tensor residual_conv2_weight,
                                           Tensor residual_conv2_bias,
                                           Tensor residual_conv3_weight,
                                           Tensor residual_conv3_bias)
        {
            input = __transformer(input,
                                  pos_embed_weight,
                                  pos_embed_bias,
                                  norm1_weight,
                                  norm1_bias,
                                  qkv_weight,
                                  proj_out_weight,
                                  proj_out_bias,
                                  norm2_weight,
                                  norm2_bias,
                                  fc1_weight,
                                  fc1_bias,
                                  fc2_weight,
                                  fc2_bias);
            input = __residual_block(input,
                                     residual_conv1_weight,
                                     residual_conv1_bias,
                                     residual_conv2_weight,
                                     residual_conv2_bias,
                                     residual_conv3_weight,
                                     residual_conv3_bias);
            return input;
        }

        private static Tensor __conv2d_k3s1_cat(Tensor input0, Tensor input1, Tensor weight, Tensor bias)
        {
            int height = input0.shape_ptr[1];
            int width = input0.shape_ptr[2];
            int src_c0 = input0.shape_ptr[0];
            int src_c1 = input1.shape_ptr[0];
            int src_c = src_c0 + src_c1;
            int dst_c = weight.shape_ptr[0];
            Tensor output = new Tensor(dst_c, height, width);
            float* src0 = input0.data_ptr;
            float* src1 = input1.data_ptr;
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
                    for(int sc = 0; sc < src_c0; ++sc)
                    {
                        float* src01 = src0 + sc * height * width;
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
                            float* src02 = src01 + sy * width;
                            for(int kx = -1; kx < 2; ++kx)
                            {
                                int sx = dx + kx;
                                if((sx >= 0) && (sx < width))
                                {
                                    *buffer++ = src02[sx];
                                }
                                else
                                {
                                    *buffer++ = 0f;
                                }
                            }
                        }
                    }
                    for(int sc = 0; sc < src_c1; ++sc)
                    {
                        float* src11 = src1 + sc * height * width;
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
                            float* src12 = src11 + sy * width;
                            for(int kx = -1; kx < 2; ++kx)
                            {
                                int sx = dx + kx;
                                if((sx >= 0) && (sx < width))
                                {
                                    *buffer++ = src12[sx];
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
            GC.KeepAlive(input0);
            GC.KeepAlive(input1);
            return output;
        }

        public static Tensor __upsampling_lrelu(Tensor input,
                                                Tensor conv1_weight,
                                                Tensor conv1_bias,
                                                Tensor conv2_weight,
                                                Tensor conv2_bias)
        {
            int src_c = input.shape_ptr[0];
            int src_h = input.shape_ptr[1];
            int src_w = input.shape_ptr[2];
            int dst_h = src_h * 4;
            int dst_w = src_w * 4;
            Tensor output = new Tensor(src_c, dst_h, dst_w);
            float* src = input.data_ptr;
            float* w1 = conv1_weight.data_ptr;
            float* b1 = conv1_bias.data_ptr;
            float* w2 = conv2_weight.data_ptr;
            float* b2 = conv2_bias.data_ptr;
            float* dst = output.data_ptr;
            Parallel.For(0, src_h, (int sy) =>
            {
                float* buffer0 = stackalloc float[src_c];
                float* buffer1 = stackalloc float[src_c * 4];
                float* buffer2 = stackalloc float[src_c * 16];
                for(int sx = 0; sx < src_w; ++sx)
                {
                    for(int sc = 0; sc < src_c; ++sc)
                    {
                        buffer0[sc] = src[(sc * src_h + sy) * src_w + sx];
                    }
                    for(int dc = 0; dc < src_c * 4; ++dc)
                    {
                        float sum = 0f;
                        for(int sc = 0; sc < src_c; ++sc)
                        {
                            sum += buffer0[sc] * w1[dc * src_c + sc];
                        }
                        buffer1[dc] = sum + b1[dc];
                    }
                    for(int p = 0; p < 4; ++p)
                    {
                        for(int dc = 0; dc < src_c * 4; ++dc)
                        {
                            float sum = 0f;
                            for(int sc = 0; sc < src_c; ++sc)
                            {
                                sum += buffer1[sc * 4 + p] * w2[dc * src_c + sc];
                            }
                            buffer2[p * src_c * 4 + dc] = sum + b2[dc];
                        }
                    }
                    for(int ky1 = 0; ky1 < 2; ++ky1)
                    {
                        int dy1 = sy * 2 + ky1;
                        for(int kx1 = 0; kx1 < 2; ++kx1)
                        {
                            int dx1 = sx * 2 + kx1;
                            for(int ky2 = 0; ky2 < 2; ++ky2)
                            {
                                int dy2 = dy1 * 2 + ky2;
                                for(int kx2 = 0; kx2 < 2; ++kx2)
                                {
                                    int dx2 = dx1 * 2 + kx2;
                                    for(int dc = 0; dc < src_c; ++dc)
                                    {
                                        float val = buffer2[(ky1 * 2 + kx1) * src_c * 4 + (dc * 2 + ky2) * 2 + kx2];
                                        if(val < 0f)
                                        {
                                            val *= 0.2f;
                                        }
                                        dst[(dc * dst_h + dy2) * dst_w + dx2] = val;
                                    }
                                }
                            }
                        }
                    }
                }
            });
            GC.KeepAlive(input);
            return output;
        }

        private static Tensor __bilinear_interpolate_add__(Tensor residual, Tensor low_res, int scale_factor)
        {
            int channels = low_res.shape_ptr[0];
            int src_h = low_res.shape_ptr[1];
            int src_w = low_res.shape_ptr[2];
            int dst_h = src_h * scale_factor;
            int dst_w = src_w * scale_factor;
            float* src = low_res.data_ptr;
            float* dst = residual.data_ptr;
            for(int c = 0; c < channels; ++c)
            {
                for(int dy = 0; dy < dst_h; ++dy)
                {
                    float sy = (dy + 0.5f) / scale_factor - 0.5f;
                    int y1 = (int)Math.Floor(sy);
                    int y1_ = Math.Max(Math.Min(src_h - 1, y1), 0);
                    int y2 = y1 + 1;
                    int y2_ = Math.Max(Math.Min(src_h - 1, y2), 0);
                    for(int dx = 0; dx < dst_w; ++dx)
                    {
                        float sx = (dx + 0.5f) / scale_factor - 0.5f;
                        int x1 = (int)Math.Floor(sx);
                        int x1_ = Math.Max(Math.Min(src_w - 1, x1), 0);
                        int x2 = x1 + 1;
                        int x2_ = Math.Max(Math.Min(src_w - 1, x2), 0);
                        float p11 = src[(c * src_h + y1_) * src_w + x1_];
                        float p12 = src[(c * src_h + y2_) * src_w + x1_];
                        float p21 = src[(c * src_h + y1_) * src_w + x2_];
                        float p22 = src[(c * src_h + y2_) * src_w + x2_];
                        dst[(c * dst_h + dy) * dst_w + dx] += (p11 * (x2 - sx) * (y2 - sy) +
                                                               p12 * (x2 - sx) * (sy - y1) +
                                                               p21 * (sx - x1) * (y2 - sy) +
                                                               p22 * (sx - x1) * (sy - y1)) / ((x2 - x1) * (y2 - y1));
                    }
                }
            }
            GC.KeepAlive(low_res);
            return residual;
        }

        public Bitmap Process(Bitmap src, Action<int> state)
        {
            state(0);
            Tensor x = __bitmap2tensor(src);
            state(1);
            Tensor x0 = __conv2d_k3s1(x, this.__head_weight, this.__head_bias);
            state(13); // 12.25 + 1
            Tensor body = __base_block(x0,
                                       this.__body_layers_0_0_pos_embed_weight,
                                       this.__body_layers_0_0_pos_embed_bias,
                                       this.__body_layers_0_0_norm1_weight,
                                       this.__body_layers_0_0_norm1_bias,
                                       this.__body_layers_0_0_attn_qkv_weight,
                                       this.__body_layers_0_0_attn_proj_out_weight,
                                       this.__body_layers_0_0_attn_proj_out_bias,
                                       this.__body_layers_0_0_norm2_weight,
                                       this.__body_layers_0_0_norm2_bias,
                                       this.__body_layers_0_0_mlp_fc_0_weight,
                                       this.__body_layers_0_0_mlp_fc_0_bias,
                                       this.__body_layers_0_0_mlp_fc_2_weight,
                                       this.__body_layers_0_0_mlp_fc_2_bias,
                                       this.__body_layers_0_1_net_0_weight,
                                       this.__body_layers_0_1_net_0_bias,
                                       this.__body_layers_0_1_net_2_weight,
                                       this.__body_layers_0_1_net_2_bias,
                                       this.__body_layers_0_1_net_4_weight,
                                       this.__body_layers_0_1_net_4_bias);
            state(26);
            body = __base_block(body,
                                this.__body_layers_1_0_pos_embed_weight,
                                this.__body_layers_1_0_pos_embed_bias,
                                this.__body_layers_1_0_norm1_weight,
                                this.__body_layers_1_0_norm1_bias,
                                this.__body_layers_1_0_attn_qkv_weight,
                                this.__body_layers_1_0_attn_proj_out_weight,
                                this.__body_layers_1_0_attn_proj_out_bias,
                                this.__body_layers_1_0_norm2_weight,
                                this.__body_layers_1_0_norm2_bias,
                                this.__body_layers_1_0_mlp_fc_0_weight,
                                this.__body_layers_1_0_mlp_fc_0_bias,
                                this.__body_layers_1_0_mlp_fc_2_weight,
                                this.__body_layers_1_0_mlp_fc_2_bias,
                                this.__body_layers_1_1_net_0_weight,
                                this.__body_layers_1_1_net_0_bias,
                                this.__body_layers_1_1_net_2_weight,
                                this.__body_layers_1_1_net_2_bias,
                                this.__body_layers_1_1_net_4_weight,
                                this.__body_layers_1_1_net_4_bias);
            state(38);
            body = __base_block(body,
                                this.__body_layers_2_0_pos_embed_weight,
                                this.__body_layers_2_0_pos_embed_bias,
                                this.__body_layers_2_0_norm1_weight,
                                this.__body_layers_2_0_norm1_bias,
                                this.__body_layers_2_0_attn_qkv_weight,
                                this.__body_layers_2_0_attn_proj_out_weight,
                                this.__body_layers_2_0_attn_proj_out_bias,
                                this.__body_layers_2_0_norm2_weight,
                                this.__body_layers_2_0_norm2_bias,
                                this.__body_layers_2_0_mlp_fc_0_weight,
                                this.__body_layers_2_0_mlp_fc_0_bias,
                                this.__body_layers_2_0_mlp_fc_2_weight,
                                this.__body_layers_2_0_mlp_fc_2_bias,
                                this.__body_layers_2_1_net_0_weight,
                                this.__body_layers_2_1_net_0_bias,
                                this.__body_layers_2_1_net_2_weight,
                                this.__body_layers_2_1_net_2_bias,
                                this.__body_layers_2_1_net_4_weight,
                                this.__body_layers_2_1_net_4_bias);
            state(50);
            body = __base_block(body,
                                this.__body_layers_3_0_pos_embed_weight,
                                this.__body_layers_3_0_pos_embed_bias,
                                this.__body_layers_3_0_norm1_weight,
                                this.__body_layers_3_0_norm1_bias,
                                this.__body_layers_3_0_attn_qkv_weight,
                                this.__body_layers_3_0_attn_proj_out_weight,
                                this.__body_layers_3_0_attn_proj_out_bias,
                                this.__body_layers_3_0_norm2_weight,
                                this.__body_layers_3_0_norm2_bias,
                                this.__body_layers_3_0_mlp_fc_0_weight,
                                this.__body_layers_3_0_mlp_fc_0_bias,
                                this.__body_layers_3_0_mlp_fc_2_weight,
                                this.__body_layers_3_0_mlp_fc_2_bias,
                                this.__body_layers_3_1_net_0_weight,
                                this.__body_layers_3_1_net_0_bias,
                                this.__body_layers_3_1_net_2_weight,
                                this.__body_layers_3_1_net_2_bias,
                                this.__body_layers_3_1_net_4_weight,
                                this.__body_layers_3_1_net_4_bias);
            state(62);
            body = __base_block(body,
                                this.__body_layers_4_0_pos_embed_weight,
                                this.__body_layers_4_0_pos_embed_bias,
                                this.__body_layers_4_0_norm1_weight,
                                this.__body_layers_4_0_norm1_bias,
                                this.__body_layers_4_0_attn_qkv_weight,
                                this.__body_layers_4_0_attn_proj_out_weight,
                                this.__body_layers_4_0_attn_proj_out_bias,
                                this.__body_layers_4_0_norm2_weight,
                                this.__body_layers_4_0_norm2_bias,
                                this.__body_layers_4_0_mlp_fc_0_weight,
                                this.__body_layers_4_0_mlp_fc_0_bias,
                                this.__body_layers_4_0_mlp_fc_2_weight,
                                this.__body_layers_4_0_mlp_fc_2_bias,
                                this.__body_layers_4_1_net_0_weight,
                                this.__body_layers_4_1_net_0_bias,
                                this.__body_layers_4_1_net_2_weight,
                                this.__body_layers_4_1_net_2_bias,
                                this.__body_layers_4_1_net_4_weight,
                                this.__body_layers_4_1_net_4_bias);
            state(75);
            x0 = __conv2d_k3s1_cat(x0, body, this.__fuse_weight, this.__fuse_bias);
            state(87);
            x0 = __upsampling_lrelu(x0,
                                    this.__upsapling_0_weight,
                                    this.__upsapling_0_bias,
                                    this.__upsapling_2_weight,
                                    this.__upsapling_2_bias);
            x0 = __conv2d_k3s1(x0, this.__tail_weight, this.__tail_bias);
            x = __bilinear_interpolate_add__(x0, x, 4);
            state(99);
            Bitmap result = __tensor2bitmap(x);
            state(100);
            return result;
        }

        #region Singleton

        public static readonly SRViT Instance;

        static SRViT()
        {
            Instance = new SRViT();
        }

        #endregion

    }

}