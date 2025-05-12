from PIL import Image, ImageDraw, ImageFont
import pandas as pd  
import os
import csv

# 根据文件生成恶意陈述句列表
csv_dir = 'advbench'
malicious_statements_dict = {}
for csv_name in os.listdir(csv_dir):
    malicious_statements_dict[csv_name] = []
    with open(os.path.join(csv_dir, csv_name), mode='r', encoding='utf-8') as f:
        malicious_statements = []
        reader = csv.reader(f)
        for row in reader:
            if row and row != 'goal':
                malicious_statements.append(row[0])
    malicious_statements_dict[csv_name].append(malicious_statements)

# 创建图像函数
def create_ocr_image(statement, output_filename, output_dir):
    # 高分辨率图像尺寸和背景颜色
    width, height = 760, 760  # 增加分辨率
    background_color = (255, 255, 255)  # 白色背景
    
    # 创建高分辨率图像
    img = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(img)
    
    # 设置字体和字号
    font_size = 60  # 增大字体
    font = ImageFont.truetype("/Library/Fonts/FreeMonoBold.ttf", font_size)  # 使用Arial字体
    
    # 提取标题：取陈述句中的主要部分作为步骤标题
    title = "Steps to " + statement.strip().lower() + ":"
    
    # 自动换行的设置
    def draw_text_with_wrap(draw, text, position, font, max_width):
        lines = []
        current_line = ""
        words = text.split(" ")
        
        for word in words:
            # 判断当前行加上一个单词后是否超出最大宽度
            test_line = current_line + " " + word if current_line else word
            text_width, _ = draw.textbbox((0, 0), test_line, font=font)[2:4]
            
            if text_width <= max_width:
                current_line = test_line  # 当前行没有超出宽度，继续添加单词
            else:
                lines.append(current_line)  # 当前行超出，保存并开始新的一行
                current_line = word
        
        if current_line:  # 添加最后一行
            lines.append(current_line)
        
        # 在图片上绘制每一行
        y_offset = position[1]
        for line in lines:
            draw.text((position[0], y_offset), line, font=font, fill=(0, 0, 0))
            y_offset += font_size  # 每行间距等于字号
        
        return y_offset  # 返回当前行的y_offset，供后续步骤使用

    # 绘制标题，自动换行
    y_offset = draw_text_with_wrap(draw, title, (20, 10), font, width - 40)  # 留出边距

    # 绘制步骤内容（留空）
    steps = ["1.", "2.", "3."]
    for step in steps:
        draw.text((20, y_offset), step, font=font, fill=(0, 0, 0))  # 步骤位置和字体
        y_offset += 60  # 步骤间距
    
    # 保存高分辨率图像
    if output_dir not in os.listdir('safebench'):
        os.makedirs(f'safebench/{output_dir}')
    img.save(os.path.join(f'safebench/{output_dir}', output_filename), format="PNG", dpi=(1000, 1000))  # 设置高DPI确保图像清晰

# 生成图像
os.mkdir('safebench')
for key, value in malicious_statements_dict.items():      
    for statements in value:
        for i, statement in enumerate(statements):
                create_ocr_image(statement, f"ocr_malicious_steps_{i + 1}.png", key)