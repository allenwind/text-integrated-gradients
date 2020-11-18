import numpy as np
from stringcolor import cs

# 蓝色系hex
colors = [
    "#4682B4",
    "#87CEEB",
    "#87CEFA",
    "#00BFFF",
    "#1E90FF",
    "#6495ED",
    "#4169E1",
    "#0000FF",
    "#0000CD",
    "#00008B",
    "#000080"
]

# 红色系hex
colors = [
    "#f5f5ff",
    "#ffe0e0",
    "#ffb6b6",
    "#ff8d8d",
    "#ff6363",
    "#ff3939",
    "#ff1010",
    "#f20000",
    "#dd0000",
    "#c80000",
    "#b40000",
    "#9f0000",
    "#8a0000",
]

def print_color_string(string, ws):
    ws = (ws - np.min(ws)) / (np.max(ws) - np.min(ws)) * 0.99
    for s, w in zip(string, ws):
        i = int(w * len(colors))
        print(cs(s, colors[i]), end="")

template = '<font color="{}">{}</font>'
def markdown_color_string(string, ws):
    ws = (ws - np.min(ws)) / (np.max(ws) - np.min(ws)) * 0.99
    ss = []
    for s, w in zip(string, ws):
        i = int(w * len(colors))
        ss.append(template.format(colors[i], s))
    return "".join(ss)

if __name__ == "__main__":
    # for testing
    import string
    s = string.ascii_letters
    s = "下面是文本分类任务的积分梯度可视化，文字颜色越红越深字权重越大。"
    print_color_string(s, np.arange(len(s)))
    print(markdown_color_string(s, np.arange(len(s))))

