import random
import string
import os
import glob
import argparse

def generate_haystack_prompt(
    total_length=2048,
    needle="这不是重点",
    hay_characters=string.ascii_letters + string.digits + "，。！？；：",
    needle_position=None
):
    """
    生成一个大海捞针prompt，needle只出现一次，其余为随机干扰字符。
    :param total_length: 生成的总长度
    :param needle: 目标短语
    :param hay_characters: 干扰字符集
    :param needle_position: 针出现的位置（None为随机）
    :return: prompt字符串
    """
    if total_length < len(needle):
        raise ValueError("总长度不能小于needle长度")
    if needle_position is None:
        needle_position = random.randint(0, total_length - len(needle))
    elif not (0 <= needle_position <= total_length - len(needle)):
        raise ValueError("needle_position超出范围")

    # 生成干扰内容
    haystack = [
        random.choice(hay_characters)
        for _ in range(total_length - len(needle))
    ]
    # 插入needle
    prompt = ''.join(haystack[:needle_position]) + needle + ''.join(haystack[needle_position:])
    return prompt

def collect_code_files(extensions=('.cpp', '.c', '.h', '.hpp'), 
                      dirs=('src', 'include', 'ggml/src', 'examples', 'tools'),
                      max_length=500000,
                      max_files=None,
                      exclude_dirs=None,
                      verbose=True):
    """
    从仓库中收集代码文件内容
    :param extensions: 要收集的文件扩展名
    :param dirs: 要搜索的目录
    :param max_length: 最大字符数
    :param max_files: 最大文件数量
    :param exclude_dirs: 要排除的目录
    :param verbose: 是否打印详细信息
    :return: 拼接后的代码文本和总字符数
    """
    all_content = []
    total_chars = 0
    file_count = 0
    
    if exclude_dirs is None:
        exclude_dirs = []
    
    # 找到所有符合条件的文件路径
    all_files = []
    for directory in dirs:
        if not os.path.exists(directory):
            if verbose:
                print(f"目录不存在: {directory}")
            continue
            
        for ext in extensions:
            pattern = os.path.join(directory, f'**/*{ext}')
            for file_path in glob.glob(pattern, recursive=True):
                # 检查是否在排除目录中
                if any(exclude_dir in file_path for exclude_dir in exclude_dirs):
                    continue
                all_files.append(file_path)
    
    # 随机打乱文件顺序
    random.shuffle(all_files)
    
    # 读取文件内容
    for file_path in all_files:
        if max_files is not None and file_count >= max_files:
            break
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                file_header = f"\n\n----- FILE: {file_path} -----\n\n"
                
                # 检查添加这个文件是否会超出最大长度限制
                remaining_chars = max_length - total_chars if max_length > 0 else float('inf')
                
                if len(file_header) + len(content) > remaining_chars:
                    # 需要截断文件
                    if remaining_chars > len(file_header) + 100:  # 确保至少有100个字符的内容
                        truncated_content = content[:remaining_chars - len(file_header)]
                        truncated_content += "\n\n... (文件内容已截断) ..."
                        
                        all_content.append(file_header)
                        all_content.append(truncated_content)
                        total_chars += len(file_header) + len(truncated_content)
                        file_count += 1
                        
                        if verbose:
                            print(f"添加文件(已截断): {file_path}, 当前总字符数: {total_chars}, 文件数: {file_count}")
                        break  # 达到最大长度，退出循环
                    else:
                        # 剩余空间太小，无法添加有意义的内容
                        if verbose:
                            print(f"跳过文件: {file_path}, 剩余空间不足")
                        continue
                else:
                    # 完整添加文件
                    all_content.append(file_header)
                    all_content.append(content)
                    total_chars += len(file_header) + len(content)
                    file_count += 1
                    
                    if verbose:
                        print(f"添加文件: {file_path}, 当前总字符数: {total_chars}, 文件数: {file_count}")
                    
                    if max_length > 0 and total_chars >= max_length:
                        break  # 达到最大长度，退出循环
        except Exception as e:
            if verbose:
                print(f"读取文件 {file_path} 出错: {e}")
    
    return ''.join(all_content), total_chars, file_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成长代码提示文件')
    parser.add_argument('--output', type=str, default="prompt.txt", help='输出文件名')
    parser.add_argument('--length', type=int, default=150000, help='最大字符数')
    parser.add_argument('--files', type=int, default=None, help='最大文件数')
    parser.add_argument('--magic', type=str, default="18810918163", help='隐藏的magic number')
    parser.add_argument('--question', type=str, default="请找出隐藏在代码中的magic number是多少?", 
                        help='要问的问题')
    parser.add_argument('--dirs', type=str, nargs='+', 
                      default=['src', 'include', 'ggml/src', 'examples', 'tools'],
                      help='要搜索的目录列表')
    parser.add_argument('--exclude', type=str, nargs='+', default=[], 
                      help='要排除的目录列表')
    parser.add_argument('--extensions', type=str, nargs='+', 
                      default=['.cpp', '.c', '.h', '.hpp'], 
                      help='要包含的文件扩展名')
    parser.add_argument('--no-shuffle', action='store_true', 
                      help='不随机打乱文件顺序')
    parser.add_argument('--haystack', action='store_true',
                      help='使用随机字符而不是代码文件')
    
    args = parser.parse_args()
    
    if not args.no_shuffle:
        random.seed()
    
    if args.haystack:
        # 使用原始的随机字符方法
        magic_number = args.magic
        prompt = generate_haystack_prompt(total_length=args.length, needle=f"the magic number is '{magic_number}'")
        prompt += f"\n\n{args.question}"
        
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"已生成随机字符文件 {args.output}，总长度：{len(prompt)} 字符")
    else:
        # 收集代码文件
        code_content, content_length, file_count = collect_code_files(
            extensions=args.extensions,
            dirs=args.dirs,
            max_length=args.length,
            max_files=args.files,
            exclude_dirs=args.exclude
        )
        
        # 添加needle和问题
        needle = f"the magic number is '{args.magic}'"
        needle_position = random.randint(0, len(code_content))
        
        # 插入needle
        final_content = code_content[:needle_position] + "\n\n" + needle + "\n\n" + code_content[needle_position:]
        final_content += f"\n\n{args.question}"
        
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(final_content)
        print(f"已生成 {args.output}，总长度：{len(final_content)} 字符，包含 {file_count} 个文件")