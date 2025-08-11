import argparse
from pathlib import Path
import os
import yaml

from ..tools.mcp.mcp_server import main as mcp_main

def start_mcp():
    """主函数 - 命令行启动入口"""
    parser = argparse.ArgumentParser(description="启动VASP MCP服务器")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--port", type=int, default=8933, help="服务器端口")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--work-dir", default=f".", help="工作目录")
    parser.add_argument("--debug", action="store_true", help="开启调试模式")
    
    args = parser.parse_args()
    
    if not args.config:
        print(f"❌ 请用 --config 设置配置文件路径")

    # 处理路径
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = os.path.abspath(config_path)
    
    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    
    # 检查配置文件（如果需要的话）
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    print(f"🚀 启动VASP MCP服务器...")
    print(f"📁 工作目录: {work_dir}")
    
    # 启动MCP服务器
    mcp_main(config_path=config_path, port=args.port, host=args.host)


if __name__ == "__main__":
    start_mcp()
