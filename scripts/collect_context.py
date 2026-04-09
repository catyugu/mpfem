import os
import argparse
from pathlib import Path

# 定义需要忽略的文件夹和文件
IGNORE_DIRS = {
    '.git', '.vscode', '.idea', '__pycache__', 'node_modules', 
    'venv', 'env', 'dist', 'build', 'target', '.next', '.pytest_cache'
}
IGNORE_FILES = {
    '.DS_Store', 'package-lock.json', 'yarn.lock', 'poetry.lock', 
    'pnpm-lock.yaml', 'favicon.ico'
}
# 定义允许读取的文本文件后缀 (可以根据需要添加)
ALLOWED_EXTENSIONS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.c', '.cpp', '.java', 
    '.go', '.rs', '.php', '.rb', '.h', '.hpp', '.sql', '.yaml', 
    '.yml', '.json', '.md', '.txt', '.html', '.css', '.sh', '.ini', '.conf'
}

def is_text_file(file_path):
    """判断是否为应读取的文本文件"""
    return file_path.suffix.lower() in ALLOWED_EXTENSIONS

def generate_tree(root_dir, prefix=""):
    """递归生成目录树结构字符串"""
    tree_str = ""
    paths = sorted(list(root_dir.iterdir()), key=lambda x: (x.is_file(), x.name))
    
    # 过滤掉忽略的项目
    paths = [p for p in paths if p.name not in IGNORE_DIRS and p.name not in IGNORE_FILES]
    
    for i, path in enumerate(paths):
        connector = "└── " if i == len(paths) - 1 else "├── "
        tree_str += f"{prefix}{connector}{path.name}\n"
        if path.is_dir():
            extension = "    " if i == len(paths) - 1 else "│   "
            tree_str += generate_tree(path, prefix + extension)
    return tree_str

def process_repository(repo_path, output_file):
    root_path = Path(repo_path).resolve()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 1. 写入项目标题
        f.write(f"# Project Source Code: {root_path.name}\n\n")
        
        # 2. 写入目录树
        f.write("## Directory Structure\n")
        f.write("```text\n")
        f.write(".\n")
        f.write(generate_tree(root_path))
        f.write("```\n\n")
        
        # 3. 递归遍历并写入文件内容
        f.write("## File Contents\n\n")
        for current_path, dirs, files in os.walk(root_path):
            # 原地修改 dirs 以便 os.walk 跳过忽略的目录
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                if file in IGNORE_FILES:
                    continue
                
                file_path = Path(current_path) / file
                if is_text_file(file_path):
                    relative_path = file_path.relative_to(root_path)
                    
                    f.write(f"### File: {relative_path}\n")
                    # 根据后缀选择 Markdown 语言标识符
                    lang = file_path.suffix.lstrip('.')
                    f.write(f"```{lang}\n")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as code_f:
                            f.write(code_f.read())
                    except Exception as e:
                        f.write(f"/* Error reading file: {e} */")
                    
                    f.write("\n```\n\n")

    print(f"✅ 处理完成！输出文件已保存至: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将代码仓库整理为 AI 友好的 Markdown 格式")
    parser.add_argument("input_folder", help="要读取的文件夹路径")
    parser.add_argument("output_file", help="输出的文本文件名 (例如 output.md)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_folder):
        print(f"❌ 错误: 文件夹 '{args.input_folder}' 不存在")
    else:
        process_repository(args.input_folder, args.output_file)