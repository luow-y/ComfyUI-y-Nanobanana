\# ComfyUI-y-Nanobanana

ComfyUI扩展节点，通过NunoBanana的API实现图像生成，提供文生图(Text-to-Image)和图生图(Image-to-Image)功能，内置授权验证系统和多种标准画布尺寸预设


ui展示


<img width="1681" height="949" alt="image" src="https://github.com/user-attachments/assets/bc8b80c5-d7c6-4004-a2bb-b3cbcdbf82a0" />


\## 📞 联系作者


<img width="526" height="823" alt="image" src="https://github.com/user-attachments/assets/f41f24de-55e2-4545-965d-59ce556fa781" />

\## 🎨 功能特点
\- ✨ \*\*双模式生成\*\*：支持文生图和图生图

\- 📏 \*\*多种尺寸\*\*：内置8种标准画布尺寸预设

\- 🔐 \*\*授权验证\*\*：安全的授权码验证系统

\- ⚡ \*\*智能检测\*\*：自动识别输入类型，优化生成效果

\- 🌐 \*\*稳定连接\*\*：多域名容错，保证服务稳定性



\## 🚀 安装方法

1，将节点文件下载到你的 ComfyUI 的 custom\_nodes 目录

git clone https://github.com/luow-y/ComfyUI-y-Nanobanana.git



2，安装所需依赖


cd ComfyUI-y-Nanobanana（去到节点的目录下）


pip install -r requirements.txt(安装依赖)



重启ComfyUI即可使用。
  
\## 📋 使用方法


\### 1. 获取授权码
联系作者微信：`ddwei089` 获取授权码



\### 2. 基础使用

1\. 在ComfyUI中搜索添加"comfyui-y-NanoBanana"节点

2\. 输入授权码和提示词

3\. 选择生成模式（文生图/图生图）

4\. 点击生成


\### 3. 设置画布尺寸（可选）

1\. 添加"NanoBanana图像尺寸"节点

2\. 选择需要的尺寸预设

3\. 连接到主节点的"尺寸"输入



\## 📐 支持的画布尺寸



| 比例 | 尺寸 | 用途 |
|------|------|------|
| 1:1 | 1024x1024 | 头像、Logo |
| 3:4 | 896x1152 | 人像摄影 |
| 5:8 | 832x1216 | 海报设计 |
| 9:16 | 768x1344 | 手机壁纸 |
| 16:9 | 1344x768 | 横版海报 |



\## ⚠️ 注意事项


\- 首次使用需要联系作者获取授权码

\- 请确保网络连接稳定

\- 建议使用标准尺寸预设以获得最佳效果



\## 🐛 问题反馈


如果遇到问题，请：

1\. 检查授权码是否正确

2\. 确认网络连接正常

3\. 查看ComfyUI控制台错误信息

4\. 联系作者微信：`ddwei089`








\## 📝 更新日志


\### v1.0.0

\- 初始版本发布

\- 支持文生图和图生图功能

\- 内置授权验证系统

\- 提供多种画布尺寸预设







如果这个节点对你有帮助，欢迎给个Star⭐支持一下！

