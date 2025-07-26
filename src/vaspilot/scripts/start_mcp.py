import argparse
from pathlib import Path

import yaml

from ..tools.mcp.mcp_server import main as mcp_main

def start_mcp():
    """主函数 - 命令行启动入口"""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent        # 项目根目录
    
    parser = argparse.ArgumentParser(description="启动VASP MCP服务器")
    parser.add_argument("--config", default=f"{project_root}/configs/mcp_config.yaml", help="配置文件路径")
    parser.add_argument("--port", type=int, default=8933, help="服务器端口")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--work-dir", default=f".", help="工作目录")
    parser.add_argument("--debug", action="store_true", help="开启调试模式")
    
    args = parser.parse_args()
    
    # 处理路径
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / args.config
    
    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = current_dir / args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    work_dir = str(work_dir.absolute())
    
    # 检查配置文件（如果需要的话）
    if config_path.exists():
        with open(config_path, "r", encoding='utf-8') as f:
            crew_config = yaml.load(f, Loader=yaml.FullLoader)
        print(f"✅ 已加载配置文件: {config_path}")
    else:
        print(f"⚠️  配置文件不存在，使用默认配置: {config_path}")
    
    print(f"🚀 启动VASP MCP服务器...")
    print(f"📁 工作目录: {work_dir}")
    
    # 启动MCP服务器
    mcp_main(config_path=config_path, port=args.port, host=args.host)


if __name__ == "__main__":
    start_mcp()
