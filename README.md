<p align="center">
  <img width="300" height="100" src="https://github.com/ColorfulSoft/ReactSR/blob/main/.content/ReactSR.png">
</p>

# ReactSR v2.0
AI-based real world super resolution application for React OS, Windows and Linux

# Supported platforms

* ReactSR supports Windows (XP and later), Linux (tested on current versions of Mint, Ubuntu, RedOS, Alt Linux and Astra Linux) and ReactOS. macOS and Android __ARE NOT SUPPORTED__.
* ReactSR runs on Windows via pre-installed .NET Framework, on ReactOS via manually installed .NET Framework 4.0, on other systems you should install Mono runtime manually.
* x86, x86-64, Itanium and ARM processors are supported.

| Operating system | Version                                           | Runtime                                 |
|:----------------:|:-------------------------------------------------:|:---------------------------------------:|
| Windows          | XP, Vista, 7, 8, 8.1, 10, 11, Embedded Standard 7 | .NET Framework 4.0 and higer, Mono 6.12 |
| Linux            | Mint, Ubuntu, RedOS, Alt Linux, Astra Linux       | Mono 6.12                               |
| ReactOS          | 0.4.14                                            | .NET Framework 4.0                      |

# Development stack

* __OS:__ Windows 7 Ultimate x64
* __Runtime:__ .NET Framework 4.5
* __IDE:__ SharpDevelop 5.1
* __Language:__ C# 5
* __GUI framework:__ Windows Forms

# Implementation details

ReactSR v2.0 is based on _[ECCV 2024] SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for Efficient Image Super-Resolution_ network trained as real-world SR model via RealESRGAN-like pipeline. We use original MobileSR pretrained model (https://github.com/Zheng-MJ/SMFANet) to initialize parameters of our model.

<p align="center">
  <img width="600" height="390" src="https://github.com/ColorfulSoft/ReactSR/blob/main/.content/Demo.png">
</p>
