import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def force_chinese_font():
    """在conda环境中强制使用中文字体"""
    
    # 尝试直接使用字体文件路径
    font_candidates = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    ]
    
    for font_path in font_candidates:
        if os.path.exists(font_path):
            # print(f"使用字体文件: {font_path}")
            # 直接创建字体属性
            chinese_font = fm.FontProperties(fname=font_path)
            
            # 全局设置（可能不工作）
            try:
                fm.fontManager.addfont(font_path)
                font_name = chinese_font.get_name()
                plt.rcParams['font.family'] = [font_name, 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                # print(f"全局字体设置为: {font_name}")
                return chinese_font
            except:
                print("全局设置失败，使用局部设置")
                return chinese_font
    
    print("未找到中文字体文件，使用默认字体")
    return None

# 获取中文字体
chinese_font = force_chinese_font()

