"""
NanoBanana-y
"""

# 告诉ComfyUI加载 ./js 目录下的前端脚本
WEB_DIRECTORY = "./js"

print("[NanoBanana-y] 启动中...")

try:
    from .main_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    print("[NanoBanana] 节点加载成功")
    
except Exception as e:
    print(f"[NanoBanana] 加载失败: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
