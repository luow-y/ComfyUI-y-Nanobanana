import base64
import json
import requests
import numpy as np
import torch
from PIL import Image
import io
import time
import logging
import os
import hashlib
from typing import List

# 隐藏HTTP请求日志
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

# 缓存管理类
class ImageCache:
    """图像缓存管理 - 用于存储和管理尺寸模板图像"""
    
    def __init__(self):
        # 在ComfyUI的temp目录下创建缓存文件夹
        import folder_paths
        self.cache_dir = os.path.join(folder_paths.get_temp_directory(), "nanobanana_cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"[缓存] 创建缓存目录: {self.cache_dir}")
    
    def get_cache_path(self, size_key):
        """获取缓存文件路径"""
        safe_key = hashlib.md5(size_key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"size_{safe_key}.png")
    
    def has_cached_image(self, size_key):
        """检查是否已有缓存图像"""
        return os.path.exists(self.get_cache_path(size_key))
    
    def save_cached_image(self, size_key, image):
        """保存缓存图像"""
        cache_path = self.get_cache_path(size_key)
        try:
            image.save(cache_path, "PNG")
            print(f"[缓存] 保存图像: {size_key} -> {cache_path}")
            return True
        except Exception as e:
            print(f"[缓存] 保存失败: {e}")
            return False
    
    def load_cached_image(self, size_key):
        """加载缓存图像"""
        cache_path = self.get_cache_path(size_key)
        try:
            if os.path.exists(cache_path):
                image = Image.open(cache_path).convert("RGB")
                print(f"[缓存] 加载图像: {size_key}")
                return image
            return None
        except Exception as e:
            print(f"[缓存] 加载失败: {e}")
            return None
    
    def create_size_template(self, size_key, width, height):
        """创建或获取尺寸模板图像"""
        # 先尝试从缓存加载
        cached_image = self.load_cached_image(size_key)
        if cached_image:
            return cached_image
        
        # 缓存不存在，创建新图像
        print(f"[缓存] 创建新的尺寸模板: {width}x{height}")
        new_image = Image.new('RGB', (width, height), color=(255, 255, 255))
        
        # 保存到缓存
        if self.save_cached_image(size_key, new_image):
            return new_image
        else:
            # 如果保存失败，直接返回图像
            return new_image

# 创建全局缓存实例
image_cache = ImageCache()


class SimpleConfig:
    MAIN_DOMAIN_B64 = "aHR0cHM6Ly9mZmZnLnF3ZXJhczAxLnNpdGUv"
    # 备用域名 - 请替换成您的实际备用域名
    BACKUP_DOMAIN_B64 = "aHR0cHM6Ly9iYWNrdXAueW91ci1kb21haW4uY29tLw=="  # 示例: https://backup.your-domain.com/
    
    @staticmethod
    def get_api_url(endpoint_type):
        """获取API URL"""
        
        try:
            main_domain = base64.b64decode(SimpleConfig.MAIN_DOMAIN_B64).decode('utf-8').strip()
            if not main_domain.endswith('/'):
                main_domain += '/'
        except Exception as e:
            print(f"[错误] 主域名解码失败: {e}")
            return None  # 不提供默认值，避免意外连接
        
        # 构建完整URL
        if endpoint_type == "use":
            return f"{main_domain}public-api/auth-codes/use"
        elif endpoint_type == "query":
            return f"{main_domain}public-api/auth-codes/query"
        else:
            return None
    
    @staticmethod
    def get_backup_url(endpoint_type):
        """获取备用URL"""
        try:
            backup_domain = base64.b64decode(SimpleConfig.BACKUP_DOMAIN_B64).decode('utf-8').strip()
            if not backup_domain.endswith('/'):
                backup_domain += '/'
            print(f"[配置] 使用备用域名: {backup_domain}")
        except Exception as e:
            print(f"[错误] 备用域名解码失败: {e}")
            return None  # 不提供默认值
        
        if endpoint_type == "use":
            return f"{backup_domain}public-api/auth-codes/use"
        elif endpoint_type == "query":
            return f"{backup_domain}public-api/auth-codes/query"
        else:
            return None

# 简化的API客户端
class SimpleAPIClient:
    """纯自定义域名API客户端"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 10
        self.session.headers.update({
            'User-Agent': 'ComfyUI-Node/1.0',
            'Content-Type': 'application/json'
        })
    
    def make_request(self, endpoint_type, data):
        """发起API请求 - 只使用配置的自定义域名"""
        
        # 尝试主域名
        main_url = SimpleConfig.get_api_url(endpoint_type)
        if main_url:
            try:
                response = self.session.post(main_url, json=data, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        print("[API] 主域名请求成功")
                        return result
                    else:
                        print(f"[API] 主域名业务错误: {result.get('error')}")
                        return result
                else:
                    print(f"[API] 主域名HTTP错误: {response.status_code}")
            except Exception as e:
                print(f"[API] 主域名连接失败: {e}")
        
        # 尝试备用域名
        backup_url = SimpleConfig.get_backup_url(endpoint_type)
        if backup_url:
            try:
                response = self.session.post(backup_url, json=data, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        print("[API] 备用域名请求成功")
                        return result
                    else:
                        print(f"[API] 备用域名业务错误: {result.get('error')}")
                        return result
                else:
                    print(f"[API] 备用域名HTTP错误: {response.status_code}")
            except Exception as e:
                print(f"[API] 备用域名连接失败: {e}")
        
        print("[API] 所有配置的域名均无法访问")
        return {"success": False, "error": "配置的域名均无法访问"}
# 授权验证函数
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

# ComfyUI Web端点
@server.PromptServer.instance.routes.post("/nanobanana/verify")
async def web_verify_endpoint(request):
    try:
        data = await request.json()
        auth_code = data.get("auth_code", "")
        result = query_auth_code(auth_code)
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

# 通用图像处理函数
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

# 基础生成类
class BaseNanoBananaGenerator:
    """基础生成器类 - 包含共同的方法"""
    
    def __init__(self):
        self.contact = "support@example.com"  # 替换为你的联系方式
    
    def tensor_to_pils(self, tensor):
        """转换tensor到PIL图像列表 - 修复版"""
        if tensor is None: 
            return []
        
        try:
            if tensor.ndim == 3: 
                tensor = tensor.unsqueeze(0)
            
            images = []
            for i in range(tensor.shape[0]):
                array = (tensor[i].cpu().numpy() * 255.0).astype(np.uint8)
                images.append(Image.fromarray(array))
            
            return images
        except Exception as e:
            print(f"[错误] tensor_to_pils转换失败: {str(e)}")
            return []
    
    def pils_to_tensor(self, pils):
        """转换PIL图像列表到tensor - 修复版"""
        if not pils or pils is None: 
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        
        try:
            arrays = [np.array(pil.convert("RGB")).astype(np.float32) / 255.0 for pil in pils]
            tensor = torch.from_numpy(np.stack(arrays, axis=0))
            return tensor
        except Exception as e:
            print(f"[错误] pils_to_tensor转换失败: {str(e)}")
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

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
            if images and len(images) > 0:
                # 有图像输入：发送文本+图像
                content = [{"type": "text", "text": prompt}]
                # 添加图像
                for i, img in enumerate(images):
                    base64_img = self.pil_to_base64(img)
                    content.append({"type": "image_url", "image_url": {"url": base64_img}})
                print(f"[AI] {generation_mode}，发送API请求，包含 {len(images)} 张图像")
                
            else:
                # 无图像输入：纯文本模式
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

# 文生图节点 - 修改版（添加五个图像输入）
class NanoBananaTextToImage(BaseNanoBananaGenerator):
    """NanoBanana文生图节点 - 内置尺寸选择 + 五个图像输入"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "授权码": ("STRING", {"default": "", "placeholder": "请输入授权码"}),
                "提示词": ("STRING", {"multiline": True, "default": "", "placeholder": "请输入您的创意提示词（中英文都行）"}),
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

    def get_size_from_preset(self, preset):
        """从预设获取尺寸"""
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
        return preset_map.get(preset, (1024, 1024))

    def center_crop_image(self, image, target_width, target_height):
        """从中心裁剪图像到目标尺寸"""
        current_width, current_height = image.size
        
        # 如果尺寸已经匹配，直接返回
        if current_width == target_width and current_height == target_height:
            return image
        
        # 计算裁剪位置（从中心裁剪）
        left = (current_width - target_width) // 2
        top = (current_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        # 确保裁剪区域在图像范围内
        left = max(0, left)
        top = max(0, top)
        right = min(current_width, right)
        bottom = min(current_height, bottom)
        
        # 裁剪图像
        cropped_image = image.crop((left, top, right, bottom))
        
        # 如果裁剪后的尺寸仍然不匹配目标尺寸，则调整大小
        if cropped_image.size != (target_width, target_height):
            cropped_image = cropped_image.resize((target_width, target_height), Image.LANCZOS)
        
        print(f"[尺寸调整] 从 {current_width}x{current_height} 裁剪到 {target_width}x{target_height}")
        return cropped_image

    def generate(self, 授权码, 提示词, canvas_preset, **kwargs):
        """文生图生成函数 - 支持图像输入"""
        start_time = time.time()
        
        print(f"[文生图] 开始生成")
        
        # 默认返回值
        empty_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        # 收集用户输入的图像（图像1-5）
        user_images = []
        for i in range(1, 6):
            img_tensor = kwargs.get(f"图像{i}")
            if img_tensor is not None: 
                converted_images = self.tensor_to_pils(img_tensor)
                if converted_images and isinstance(converted_images, list):
                    user_images.extend(converted_images)
        
        print(f"[文生图] 收集到用户图像: {len(user_images)} 张")

        # 处理外部提示词
        external_prompt = kwargs.get("外部提示词", "")
        if external_prompt and external_prompt.strip():
            if 提示词 and 提示词.strip():
                combined_prompt = f"{提示词.strip()}\n{external_prompt.strip()}"
            else:
                combined_prompt = external_prompt.strip()
            print(f"[文生图] 使用外部提示词节点，合并后长度: {len(combined_prompt)}")
        else:
            combined_prompt = 提示词 if 提示词 else ""

        # 输入验证
        if not combined_prompt or not combined_prompt.strip():
            return (empty_tensor, "错误：请输入提示词")

        if not 授权码 or not 授权码.strip():
            return (empty_tensor, "错误：请输入授权码")

        # 获取目标尺寸
        target_width, target_height = self.get_size_from_preset(canvas_preset)
        
        # 获取或创建尺寸模板图像
        size_key = f"{canvas_preset}_{target_width}x{target_height}"
        size_template = image_cache.create_size_template(size_key, target_width, target_height)
        
        # 组合所有图像：用户图像 + 尺寸模板
        final_images = []
        if user_images:
            final_images.extend(user_images)
        # 始终添加尺寸模板图像
        final_images.append(size_template)
        
        # 根据是否有用户图像构建不同的提示词
        if user_images and len(user_images) > 0:
            # 有用户图像：图生图模式的文生图
            size_instruction = f"参考提供的图片内容和风格，在 {target_width}x{target_height} 像素的画布上生成以下描述的图像：{combined_prompt}。直接返回生成的图像，无需任何文字描述或额外说明。"
            print(f"[文生图] 图生图模式 - 用户图像: {len(user_images)} 张，尺寸: {target_width}x{target_height}")
        else:
            # 无用户图像：纯文生图模式
            size_instruction = f"请帮我在 {target_width}x{target_height} 像素的画布上生成以下描述的图像：{combined_prompt}。直接返回生成的图像，无需任何文字描述或额外说明。"
            print(f"[文生图] 纯文生图模式 - 尺寸: {target_width}x{target_height}")
        
        print(f"[文生图] 最终图像数量: {len(final_images)}，提示词长度: {len(size_instruction)}")

        # 验证授权码
        print("[文生图] 验证授权码...")
        auth_result = verify_auth_code(授权码)
        
        if not auth_result.get("success"):
            return (empty_tensor, f"验证失败: {auth_result.get('error')}")
        
        remaining = auth_result.get("remaining", "未知")
        print(f"[文生图] 验证成功，剩余: {remaining}")
        
        # 调用AI API
        result_images, msg = self.call_api(auth_result["gemini_key"], size_instruction, final_images, "文生图")
        
        total_time = time.time() - start_time
        
        if result_images and len(result_images) > 0:
            # 在输出之前检测并调整图像尺寸
            adjusted_images = []
            for i, img in enumerate(result_images):
                current_size = img.size
                if current_size != (target_width, target_height):
                    print(f"[尺寸检测] 图像 {i+1} 尺寸 {current_size} 与目标尺寸 {target_width}x{target_height} 不匹配，开始裁剪")
                    adjusted_img = self.center_crop_image(img, target_width, target_height)
                    adjusted_images.append(adjusted_img)
                else:
                    print(f"[尺寸检测] 图像 {i+1} 尺寸匹配，无需调整")
                    adjusted_images.append(img)
            
            # 成功生成图像
            out_tensor = self.pils_to_tensor(adjusted_images)
            final_msg = f"✅ {msg}，耗时 {total_time:.1f}s，剩余次数: {remaining}"
            print(f"[文生图] 成功完成: {final_msg}")
            return (out_tensor, final_msg)
        else:
            # 生成失败
            error_msg = f"❌ {msg}，耗时 {total_time:.1f}s"
            print(f"[文生图] 失败: {error_msg}")
            return (empty_tensor, error_msg)
    
    CATEGORY = "NanoBanana-y"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    FUNCTION = "generate"

# 图生图节点
class NanoBananaImageToImage(BaseNanoBananaGenerator):
    """NanoBanana图生图节点 - 使用外部尺寸节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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

    def generate(self, 授权码, 提示词, **kwargs):
        """图生图生成函数"""
        start_time = time.time()
        
        print(f"[图生图] 开始生成")
        
        # 默认返回值
        empty_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        # 第一步：收集所有图像输入
        regular_images = []
        for i in range(1, 6):
            img_tensor = kwargs.get(f"图像{i}")
            if img_tensor is not None: 
                converted_images = self.tensor_to_pils(img_tensor)
                if converted_images and isinstance(converted_images, list):
                    regular_images.extend(converted_images)
        
        size_tensor = kwargs.get("尺寸")
        size_images = []
        target_width, target_height = None, None
        if size_tensor is not None:
            size_images = self.tensor_to_pils(size_tensor)
            if not isinstance(size_images, list):
                size_images = []
            # 获取尺寸节点输入图像的尺寸
            if size_images and len(size_images) > 0:
                target_width, target_height = size_images[0].size
                print(f"[图生图] 检测到尺寸节点输入图像: {target_width}x{target_height}")
        
        # 确保所有变量都不是None
        if regular_images is None:
            regular_images = []
        if size_images is None:
            size_images = []
            
        print(f"[图生图] 图像收集完成 - 常规图像: {len(regular_images)} 张，尺寸图像: {len(size_images)} 张")

        # 处理外部提示词
        external_prompt = kwargs.get("外部提示词", "")
        if external_prompt and external_prompt.strip():
            if 提示词 and 提示词.strip():
                combined_prompt = f"{提示词.strip()}\n{external_prompt.strip()}"
            else:
                combined_prompt = external_prompt.strip()
            print(f"[图生图] 使用外部提示词节点，合并后长度: {len(combined_prompt)}")
        else:
            combined_prompt = 提示词 if 提示词 else ""

        # 图生图模式：使用所有图像
        final_images = []
        if regular_images:
            final_images.extend(regular_images)
        if size_images:
            final_images.extend(size_images)
        
        # 根据是否有尺寸图像设置不同的内置提示词
        if size_images and len(size_images) > 0:
            size_instruction = f"以 {target_width}x{target_height} 像素并参考提供的图片内容和风格，根据以下提示词生成新图片：{combined_prompt}\n，直接返回生成的图像，无需任何文字描述或额外说明。"
            final_prompt = size_instruction
            print("[图生图] 包含尺寸图像，使用像素要求提示词")
        else:
            size_instruction = f"请参考提供的图片内容和风格，根据以下提示词生成新图片：{combined_prompt}\n，直接返回生成的图像，无需任何文字描述或额外说明"
            final_prompt = size_instruction
            print("[图生图] 无尺寸图像，使用普通提示词")
        
        if not final_images or len(final_images) == 0:
            return (empty_tensor, "图生图模式需要至少一张输入图像（常规图像或尺寸图像）")

        # 输入验证
        if not final_prompt or not final_prompt.strip():
            return (empty_tensor, "错误：请输入提示词")

        if not 授权码 or not 授权码.strip():
            return (empty_tensor, "错误：请输入授权码")

        print(f"[图生图] 最终参数 - 提示词长度: {len(final_prompt) if final_prompt else 0}，图像数量: {len(final_images)}")

        # 验证授权码
        print("[图生图] 验证授权码...")
        auth_result = verify_auth_code(授权码)
        
        if not auth_result.get("success"):
            return (empty_tensor, f"验证失败: {auth_result.get('error')}")
        
        remaining = auth_result.get("remaining", "未知")
        print(f"[图生图] 验证成功，剩余: {remaining}")
        
        # 调用AI API
        result_images, msg = self.call_api(auth_result["gemini_key"], final_prompt, final_images, "图生图")
        
        total_time = time.time() - start_time
        
        if result_images and len(result_images) > 0:
            # 成功生成图像
            out_tensor = self.pils_to_tensor(result_images)
            final_msg = f"✅ {msg}，耗时 {total_time:.1f}s，剩余次数: {remaining}"
            print(f"[图生图] 成功完成: {final_msg}")
            return (out_tensor, final_msg)
        else:
            # 生成失败
            error_msg = f"❌ {msg}，耗时 {total_time:.1f}s"
            print(f"[图生图] 失败: {error_msg}")
            return (empty_tensor, error_msg)
    
    CATEGORY = "NanoBanana-y"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    FUNCTION = "generate"

# NanoBanana图像尺寸节点 - 缓存版
class NanoBananaImageSize:
    """NanoBanana图像尺寸调整 - 使用缓存的纯白画布"""
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
            # 使用缓存系统获取或创建画布
            width, height = target_size
            size_key = f"{canvas_preset}_{width}x{height}"
            canvas = image_cache.create_size_template(size_key, width, height)
            
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
        }

# 节点注册
NODE_CLASS_MAPPINGS = {
    "NanoBananaTextToImage": NanoBananaTextToImage,
    "NanoBananaImageToImage": NanoBananaImageToImage,
    "NanoBanana图像尺寸": NanoBananaImageSize,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaTextToImage": "NanoBanana文生图",
    "NanoBananaImageToImage": "NanoBanana图生图", 
    "NanoBanana图像尺寸": "NanoBanana图像尺寸",
}

# 配置检查
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
    
    # 检查缓存系统
    try:
        test_cache = image_cache.create_size_template("test_1024x1024", 1024, 1024)
        if test_cache:
            print("✅ 缓存系统正常")
        else:
            print("⚠️ 缓存系统可能有问题")
    except Exception as e:
        print(f"⚠️ 缓存系统错误: {e}")

# 运行检查
check_setup()
