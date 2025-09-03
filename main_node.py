import base64
import json
import requests
import numpy as np
import torch
from PIL import Image
import io
import time
import logging
from typing import List  # 添加这一行

# === 隐藏HTTP请求日志 ===
# 禁用requests库的HTTP请求日志
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# 为API接口导入服务器模块
import server
from aiohttp import web

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# === 简单配置类 ===
class SimpleConfig:
    MAIN_DOMAIN_B64 = "aHR0cHM6Ly9hYTExLnNtZWFscmVpYnNvbWVqdTg0LndvcmtlcnMuZGV2"
    BACKUP_DOMAIN_B64 = "aHR0cHM6Ly9mYW5jeS1wb25kLTEyYTAuc21lYWxyZWlic29tZWp1ODQud29ya2Vycy5kZXY="
    
    @staticmethod
    def get_api_url(endpoint_type):
        """获取API URL - 简单可靠"""
        
        # 解码主域名
        try:
            main_domain = base64.b64decode(SimpleConfig.MAIN_DOMAIN_B64).decode('utf-8')
        except:
            main_domain = "https://your-domain.workers.dev"  # 默认值
        
        # 构建完整URL
        if endpoint_type == "use":
            return f"{main_domain}/public-api/auth-codes/use"
        elif endpoint_type == "query":
            return f"{main_domain}/public-api/auth-codes/query"
        else:
            return None
    
    @staticmethod
    def get_backup_url(endpoint_type):
        """获取备用URL"""
        try:
            backup_domain = base64.b64decode(SimpleConfig.BACKUP_DOMAIN_B64).decode('utf-8')
        except:
            backup_domain = "https://fancy-pond-12a0.smealsomeju84.workers.dev"  # 默认备用
        
        if endpoint_type == "use":
            return f"{backup_domain}/public-api/auth-codes/use"
        elif endpoint_type == "query":
            return f"{backup_domain}/public-api/auth-codes/query"
        else:
            return None

# === 简单API客户端 ===
class SimpleAPIClient:
    """简单的API客户端 - 无复杂重试"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 10
        self.session.headers.update({
            'User-Agent': 'ComfyUI-Node/1.0',
            'Content-Type': 'application/json'
        })
    
    def make_request(self, endpoint_type, data):
        """发起API请求 - 最多2次尝试"""
        
        # 尝试主域名
        main_url = SimpleConfig.get_api_url(endpoint_type)
        if main_url:
            try:
                print(f"[API] 请求主域名...")
                response = self.session.post(main_url, json=data, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        print("[API] 主域名成功")
                        return result
                    else:
                        print(f"[API] 业务错误: {result.get('error')}")
                        return result
            except Exception as e:
                print("[API] 主域名失败")
        
        # 尝试备用域名
        backup_url = SimpleConfig.get_backup_url(endpoint_type)
        if backup_url:
            try:
                print(f"[API] 请求备用域名...")
                response = self.session.post(backup_url, json=data, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    print("[API] 备用域名成功")
                    return result
            except Exception as e:
                print("[API] 备用域名失败")
        
        return {"success": False, "error": "所有域名都无法访问"}

# === 授权验证函数 ===
def verify_auth_code(auth_code):
    """验证授权码"""
    if not auth_code or not auth_code.strip():
        return {"success": False, "error": "授权码不能为空"}
    
    client = SimpleAPIClient()
    result = client.make_request('use', {"authCode": auth_code.strip()})
    
    if result.get("success"):
        data = result.get("data", {})
        return {
            "success": True,
            "remaining": data.get("remainingRequests", 0),
            "gemini_key": data.get("geminiKey", "")
        }
    else:
        return {"success": False, "error": result.get("error", "验证失败")}

def query_auth_code(auth_code):
    """查询授权码状态"""
    if not auth_code or not auth_code.strip():
        return {"success": False, "error": "授权码不能为空"}
    
    client = SimpleAPIClient()
    result = client.make_request('query', {"authCode": auth_code.strip()})
    
    if result.get("success"):
        data = result.get("data", {})
        return {
            "success": True,
            "remaining": data.get("remainingRequests", 0),
            "status": data.get("statusText", ""),
            "usage_percent": data.get("usagePercent", "0")
        }
    else:
        return {"success": False, "error": result.get("error", "查询失败")}

# === ComfyUI Web端点 ===
@server.PromptServer.instance.routes.post("/nanobanana/verify")
async def web_verify_endpoint(request):
    try:
        data = await request.json()
        auth_code = data.get("auth_code", "")
        result = query_auth_code(auth_code)
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

# === 通用图像处理函数 ===
def _tensor_to_pils(image) -> List[Image.Image]:
    """将ComfyUI的IMAGE张量转换为PIL图像列表"""
    if isinstance(image, dict) and "images" in image:
        image = image["images"]

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"期望输入为torch.Tensor，实际得到{type(image)}")

    # 处理单张图片的情况
    if image.ndim == 3:
        image = image.unsqueeze(0)

    # 验证张量维度
    if image.ndim != 4:
        raise ValueError(f"图像张量维度错误，期望4维，实际{image.ndim}维")

    # 转换为PIL图像
    arr = (image.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)  # [B,H,W,3]
    return [Image.fromarray(arr[i], mode="RGB") for i in range(arr.shape[0])]


def _pils_to_tensor(pils: List[Image.Image]) -> torch.Tensor:
    """将PIL图像列表转换为ComfyUI的IMAGE张量"""
    if not pils:
        # 返回一个占位图像而不是空张量
        placeholder = Image.new('RGB', (64, 64), color=(255, 255, 255))
        pils = [placeholder]

    # 确保所有图像尺寸一致
    first_size = pils[0].size
    for i, pil in enumerate(pils):
        if pil.size != first_size:
            # 调整尺寸以匹配第一张图像
            pils[i] = pil.resize(first_size, Image.LANCZOS)
            print(f"警告：图像{i}尺寸{pil.size}与第一张图像{first_size}不一致，已调整尺寸")

    np_imgs = []
    for pil in pils:
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        arr = np.array(pil, dtype=np.uint8)  # [H,W,3]
        np_imgs.append(arr)

    batch = np.stack(np_imgs, axis=0).astype(np.float32) / 255.0  # [B,H,W,3]
    return torch.from_numpy(batch)

# === NanoBanana图像尺寸节点 - 修复版 ===
class NanoBananaImageSize:
    """NanoBanana图像尺寸调整 - 直接生成标准尺寸的白底画布"""
    CATEGORY = "NanoBanana-y"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "canvas_preset": ([
                    "1:1 - 1024x1024",
                    "3:4 - 896x1152", 
                    "5:8 - 832x1216",
                    "9:16 - 768x1344",
                    "9:21 - 640x1536",
                    "4:3 - 1152x896",
                    "3:2 - 1216x832", 
                    "16:9 - 1344x768",
                ], {"default": "1:1 - 1024x1024"}),
            },
        }

    def generate(self, **kwargs):
        # 获取参数
        canvas_preset = kwargs.get('canvas_preset', "1:1 - 1024x1024")

        # 尺寸配置
        preset_map = {
            "1:1 - 1024x1024": (1024, 1024),
            "3:4 - 896x1152": (896, 1152),
            "5:8 - 832x1216": (832, 1216), 
            "9:16 - 768x1344": (768, 1344),
            "9:21 - 640x1536": (640, 1536),
            "4:3 - 1152x896": (1152, 896),
            "3:2 - 1216x832": (1216, 832),
            "16:9 - 1344x768": (1344, 768),
        }

        # 获取目标尺寸
        target_size = preset_map.get(canvas_preset, (1024, 1024))
        
        try:
            # 直接生成指定尺寸的白底画布，不需要裁剪
            width, height = target_size
            canvas = Image.new('RGB', (width, height), color=(255, 255, 255))
            
            # 转换为张量
            out_tensor = _pils_to_tensor([canvas])
            
            print(f"[NanoBanana画布] 生成成功: {width}x{height}")
            return (out_tensor,)
            
        except Exception as e:
            # 出错时返回默认尺寸的白底图
            print(f"[NanoBanana画布] 生成失败: {str(e)}，返回默认尺寸")
            default_img = Image.new('RGB', (1024, 1024), color=(255, 255, 255))
            out_tensor = _pils_to_tensor([default_img])
            return (out_tensor,)

    def get_size_info(self, canvas_preset):
        """获取尺寸信息 - 可用于调试"""
        preset_map = {
            "1:1 - 1024x1024": (1024, 1024),
            "3:4 - 896x1152": (896, 1152),
            "5:8 - 832x1216": (832, 1216),
            "9:16 - 768x1344": (768, 1344), 
            "9:21 - 640x1536": (640, 1536),
            "4:3 - 1152x896": (1152, 896),
            "3:2 - 1216x832": (1216, 832),
            "16:9 - 1344x768": (1344, 768),
        }
        
        size = preset_map.get(canvas_preset, (1024, 1024))
        ratio = size[0] / size[1]
        
        return {
            "size": size,
            "width": size[0], 
            "height": size[1],
            "ratio": f"{ratio:.3f}",
            "pixels": size[0] * size[1]
        }# === 主节点类 ===
class NanoBananaAICG:
    """NanoBanana-y 生成节点 - 自动检测版"""
    
    def __init__(self):
        self.contact = "your_wechat"  # 替换为你的联系方式
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generation_mode": (["图生图 (Image-to-Image)", "文生图 (Text-to-Image)"],),
                "授权码": ("STRING", {"default": "", "placeholder": "请输入授权码"}),
                "提示词": ("STRING", {"multiline": True, "default": "", "placeholder": "请输入您的创意提示词（中英文都行）"}),
            },
            "optional": {
                "图像1": ("IMAGE",), 
                "图像2": ("IMAGE",), 
                "图像3": ("IMAGE",), 
                "图像4": ("IMAGE",), 
                "图像5": ("IMAGE",),
                "尺寸": ("IMAGE",),
                "外部提示词": ("STRING", {"forceInput": True}),
            },
        }

    def tensor_to_pils(self, tensor):
        """转换tensor到PIL图像列表"""
        if tensor is None: 
            return []
        
        if tensor.ndim == 3: 
            tensor = tensor.unsqueeze(0)
        
        images = []
        for i in range(tensor.shape[0]):
            array = (tensor[i].cpu().numpy() * 255.0).astype(np.uint8)
            images.append(Image.fromarray(array))
        
        return images
    
    def pils_to_tensor(self, pils):
        """转换PIL图像列表到tensor"""
        if not pils: 
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        
        arrays = [np.array(pil.convert("RGB")).astype(np.float32) / 255.0 for pil in pils]
        tensor = torch.from_numpy(np.stack(arrays, axis=0))
        return tensor

    def pil_to_base64(self, pil):
        """转换PIL图像到base64"""
        if pil.mode in ("RGBA", "P"): 
            pil = pil.convert("RGB")
        
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=85)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"

    def base64_to_pil(self, data):
        """转换base64到PIL图像"""
        if "base64," in data: 
            data = data.split("base64,")[1]
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")

    def call_api(self, gemini_key, prompt, images, generation_mode):
        """调用AI生成API - 自动检测图像输入"""
        if not OpenAI: 
            return [], "请安装openai库: pip install openai"
        
        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1", 
                api_key=gemini_key,
                timeout=45
            )
            
            # 根据是否有图像输入来决定API调用方式
            if images:
                # 有图像输入：发送文本+图像
                if generation_mode == "图生图 (Image-to-Image)" and not images:
                    return [], "图生图模式需要至少一张输入图像"
                
                content = [{"type": "text", "text": prompt}]
                # 添加图像
                for i, img in enumerate(images):
                    base64_img = self.pil_to_base64(img)
                    content.append({"type": "image_url", "image_url": {"url": base64_img}})
                print(f"[AI] {generation_mode}，发送API请求，包含 {len(images)} 张图像")
                
            else:
                # 无图像输入：纯文本模式
                if generation_mode == "图生图 (Image-to-Image)":
                    return [], "图生图模式需要至少一张输入图像"
                
                enhanced_prompt = f"Create a high-quality, detailed image based on this description: {prompt}"
                content = [{"type": "text", "text": enhanced_prompt}]
                print(f"[AI] {generation_mode}，发送API请求（纯文本模式）")

            response = client.chat.completions.create(
                model="google/gemini-2.5-flash-image-preview:free",
                messages=[{"role": "user", "content": content}],
                extra_headers={"HTTP-Referer": "https://comfyui.org"}
            )
            
            # 改进的响应处理
            response_dict = response.model_dump()
                       
            # 检查基本结构
            choices = response_dict.get("choices", [])
            if not choices:
                return [], "API响应中没有choices数据"
            
            message = choices[0].get("message", {})
            if not message:
                return [], "API响应中没有message数据"
            
          
            # 尝试获取图像数据
            images_list = message.get("images", [])
            
            # 如果没有images字段，检查其他可能的字段
            if not images_list:
                # 检查content字段
                content_data = message.get("content", "")
                print(f"[AI] 响应content: {str(content_data)[:500]}...")
                
                # 检查是否在content中包含图像数据
                if "data:image" in str(content_data):
                    print("[AI] 在content中发现图像数据")
                    # 尝试解析content中的base64图像
                    import re
                    image_matches = re.findall(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', str(content_data))
                    if image_matches:
                        result_images = []
                        for img_data in image_matches:
                            try:
                                result_images.append(self.base64_to_pil(img_data))
                            except Exception as e:
                                print(f"[AI] 解析content中的图像失败: {e}")
                        if result_images:
                            return result_images, f"从content中解析出 {len(result_images)} 张图像"
                
                return [], f"{generation_mode}失败：API响应中没有图像数据，可能是免费配额已用完"
            
            result_images = []
            for i, img_info in enumerate(images_list):
                img_url = img_info.get("image_url", {}).get("url", "")
                if img_url:
                    try:
                        result_images.append(self.base64_to_pil(img_url))
                    except Exception as e:
                        print(f"[AI] 解析图像 {i+1} 失败: {e}")
                        
            return result_images, f"生成 {len(result_images)} 张图像"
            
        except Exception as e:
            error_msg = str(e)
            print(f"[AI] API调用失败: {error_msg}")
            
            # 针对不同错误类型提供更具体的错误信息
            if "429" in error_msg or "rate limit" in error_msg.lower():
                return [], "速率限制：重试，不行就联系作者"
            elif "402" in error_msg or "quota" in error_msg.lower():
                return [], "配额不足：重试，不行就联系作者"
            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                return [], "授权失败：API密钥无效或已过期，重试，不行就联系作者"
            elif "timeout" in error_msg.lower():
                return [], "请求超时：网络连接不稳定，请重试"
            else:
                return [], f"{generation_mode}失败: {error_msg}"

    def generate(self, generation_mode, 授权码, 提示词, **kwargs):
        """主要生成函数"""
        start_time = time.time()
        
        auth_code = 授权码
        base_prompt = 提示词
        
        print(f"[生成] 开始，模式: {generation_mode}")
        
        # 默认返回值
        empty_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        # 第一步：收集所有图像输入
        regular_images = []
        for i in range(1, 6):
            img_tensor = kwargs.get(f"图像{i}")
            if img_tensor is not None: 
                converted_images = self.tensor_to_pils(img_tensor)
                regular_images.extend(converted_images)
        
        size_tensor = kwargs.get("尺寸")
        size_images = []
        if size_tensor is not None:
            size_images = self.tensor_to_pils(size_tensor)
        
        print(f"[生成] 图像收集完成 - 常规图像: {len(regular_images)} 张，尺寸图像: {len(size_images)} 张")

        # 第二步：处理外部提示词
        external_prompt = kwargs.get("外部提示词", "")
        if external_prompt and external_prompt.strip():
            if base_prompt.strip():
                combined_prompt = f"{base_prompt.strip()}\n{external_prompt.strip()}"
            else:
                combined_prompt = external_prompt.strip()
            print(f"[生成] 使用外部提示词节点，合并后长度: {len(combined_prompt)}")
        else:
            combined_prompt = base_prompt

        # 第三步：根据模式确定最终使用的图像和提示词
        final_images = []
        final_prompt = combined_prompt
        
        if generation_mode == "图生图 (Image-to-Image)":
            # 图生图模式：使用所有图像
            final_images.extend(regular_images)
            final_images.extend(size_images)
            
            # 如果包含尺寸图像，添加内置提示词
            if size_images:
                size_instruction = "**重要指令：请严格按照输入的白色画布尺寸进行创作，图像必须完全填满整个画布区域，不能有任何白边或留白。**\n\n请基于以下描述创作："
                final_prompt = f"{size_instruction}\n{combined_prompt.strip()}" if combined_prompt.strip() else size_instruction
                print("[生成] 图生图模式：包含尺寸图像，已添加内置提示词（置顶）")
            
            if not final_images:
                return (empty_tensor, "图生图模式需要至少一张输入图像（常规图像或尺寸图像）")
        else:
            # 文生图模式：只使用尺寸图像
            if regular_images:
                print(f"[生成] 文生图模式：忽略 {len(regular_images)} 张常规图像输入")
            
            if size_images:
                final_images.extend(size_images)
                size_instruction = "**重要指令：请严格按照输入的白色画布尺寸进行创作，图像必须完全填满整个画布区域，不能有任何白边或留白。**\n\n请基于以下描述创作："
                final_prompt = f"{size_instruction}\n{combined_prompt.strip()}" if combined_prompt.strip() else size_instruction
                print(f"[生成] 文生图模式：使用 {len(size_images)} 张尺寸图像，已添加内置提示词（置顶）")
            else:
                print("[生成] 文生图模式：纯文本生成")

        print(f"[生成] 最终参数 - 提示词长度: {len(final_prompt)}，内容预览: {final_prompt[:150]}...，图像数量: {len(final_images)}")
        print(f"[调试] 内置提示词是否生效: {'是' if '重要指令' in final_prompt else '否'}")

        # 输入验证
        if not final_prompt.strip():
            return (empty_tensor, "错误：请输入提示词")

        if not auth_code.strip():
            return (empty_tensor, "错误：请输入授权码")

        # 验证授权码
        print("[生成] 验证授权码...")
        auth_result = verify_auth_code(auth_code)
        
        if not auth_result.get("success"):
            return (empty_tensor, f"验证失败: {auth_result.get('error')}")
        
        remaining = auth_result.get("remaining", "未知")
        print(f"[生成] 验证成功，剩余: {remaining}")
        
        # 调用AI API，使用final_prompt和final_images
        result_images, msg = self.call_api(auth_result["gemini_key"], final_prompt, final_images, generation_mode)
        
        total_time = time.time() - start_time
        
        if result_images:
            tensor = self.pils_to_tensor(result_images)
            status = f"生成成功! 剩余:{remaining} 耗时:{total_time:.1f}s 生成:{len(result_images)}张"
            return (tensor, status)
        else:
            status = f"生成失败: {msg} 剩余:{remaining}"
            return (empty_tensor, status)

    CATEGORY = "NanoBanana-y"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    FUNCTION = "generate"

# === 节点注册 ===
NODE_CLASS_MAPPINGS = {
    "NanoBananaAICG": NanoBananaAICG,
    "NanoBanana图像尺寸": NanoBananaImageSize,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaAICG": "NanoBanana-y",  # 键改为匹配，显示名称自定义
    "NanoBanana图像尺寸": "NanoBanana图像尺寸",
}

# === 配置检查 ===
def check_setup():
    print("=" * 50)
    print("NanoBanana-y 节点启动检查")
    print("-" * 50)
    
    # 检查域名配置
    try:
        main_domain = base64.b64decode(SimpleConfig.MAIN_DOMAIN_B64).decode('utf-8')
        if "your-domain" in main_domain:
            print("❌ 需要配置真实域名!")
            print("请修改 SimpleConfig.MAIN_DOMAIN_B64")
            print("配置方法:")
            print("1. 运行: echo -n 'https://你的域名.workers.dev' | base64")
            print("2. 将输出结果替换 MAIN_DOMAIN_B64 的值")
        else:
            domain_name = main_domain.split('//')[1].split('.')[0]
            print(f"✅ 已配置域名: {domain_name}...")
    except Exception as e:
        print(f"❌ 域名配置错误: {e}")
    
    # 检查依赖
    missing = []
    try:
        import requests, torch, numpy, PIL
    except ImportError as e:
        missing.append(str(e))
    
    if OpenAI is None:
        missing.append("pip install openai")
    
    if missing:
        print("❌ 缺少依赖:")
        for m in missing:
            print(f"   {m}")
    else:
        print("✅ 所有依赖已安装")
    
    print("=" * 50)
    print("🔄 自动检测逻辑:")
    print("   - 检测到尺寸节点有输入时，自动启动图像处理")
    print("   - 图生图和文生图模式都会自动使用尺寸图像")
    print("   - 文生图模式下，常规图像依然被忽略")
    print(f"🔍 新增节点: NanoBanana图像尺寸 (共 {len(NODE_CLASS_MAPPINGS)} 个节点)")
    print("=" * 50)

# 运行检查
check_setup()