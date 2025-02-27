import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox
import joblib
import pandas as pd

# 加载 BP 神经网络模型 和 标准化器
bp_model = joblib.load("res/bp_model.pkl")
scaler = joblib.load("res/scaler.pkl")

# 预定义特征列
expected_columns = [
    '长表总分', '皮温mean', '△a*', '△B*', '缓慢胃率', 'PIF', 'Penh', 'SCL',
    '饮酒_2', '近视的度数_2', '近视的度数_3', '近视的度数_4',
    '成年期白天的情绪_2', '直系亲属是否有疾病史_2'
]

# 创建 ttkbootstrap 窗口
# root = ttk.Window(themename="superhero")  # 主题，可选 "darkly", "solar", "cyborg"
# root = ttk.Window(themename="darkly")  # 主题，可选 "darkly", "solar", "cyborg"
root = ttk.Window(themename="solar")  # 主题，可选 "darkly", "solar", "cyborg"
root.title("BP 神经网络预测系统")
root.geometry("700x900")  # 增大窗口尺寸

# 创建可滚动的 Frame
canvas = ttk.Canvas(root)
scrollable_frame = ttk.Frame(canvas)
scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)

canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

# 让 Frame 自动调整大小
def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

scrollable_frame.bind("<Configure>", on_frame_configure)

# 标题
title_label = ttk.Label(scrollable_frame, text="🔬 BP 神经网络预测系统", font=("Arial", 20, "bold"), bootstyle="primary")
title_label.grid(row=0, column=0, columnspan=2, pady=20)

# 存储单选选项
radio_vars = {
    "饮酒": ttk.IntVar(value=1),
    "近视的度数": ttk.IntVar(value=3),
    "成年期白天的情绪": ttk.IntVar(value=2),
    "直系亲属是否有疾病史": ttk.IntVar(value=2),
}

# 创建单选按钮
def create_radio(label, options, row):
    frame = ttk.Frame(scrollable_frame)
    frame.grid(row=row, column=0, columnspan=2, pady=5, sticky="w")

    ttk.Label(frame, text=f"{label}：", font=("Arial", 14, "bold")).pack(anchor="w", padx=10)

    for value, text in options.items():
        ttk.Radiobutton(frame, text=text, variable=radio_vars[label], value=value, bootstyle="info").pack(side="left", padx=10)

# 生成单选项
radio_options = [
    ("饮酒", {1: "不饮酒", 2: "饮酒"}),
    ("近视的度数", {1: "度数1", 2: "度数2", 3: "度数3", 4: "度数4"}),
    ("成年期白天的情绪", {1: "正常", 2: "低落"}),
    ("直系亲属是否有疾病史", {1: "无", 2: "有"}),
]

for i, (label, options) in enumerate(radio_options, start=1):
    create_radio(label, options, i)

# 存储输入框数据
entries = {}

# 创建输入框
def create_input(label, default_value, row):
    ttk.Label(scrollable_frame, text=f"{label}：", width=25, font=("Arial", 14, "bold")).grid(row=row, column=0, padx=10, pady=5, sticky="w")
    entry = ttk.Entry(scrollable_frame, font=("Arial", 12), bootstyle="info", width=15)
    entry.insert(0, str(default_value))  # 设置默认值
    entry.grid(row=row, column=1, padx=10, pady=5)
    entries[label] = entry

# 创建输入字段
input_fields = [
    ("长表总分", 75.6),
    ("皮温mean", 36.5),
    ("△a*", 0.2),
    ("△B*", 0.1),
    ("缓慢胃率", 2.3),
    ("PIF", 1.2),
    ("Penh", 0.5),
    ("SCL", 3.4),
]

for i, (label, default) in enumerate(input_fields, start=len(radio_options) + 1):
    create_input(label, default, i)

# 预测函数
def predict():
    try:
        # 获取输入值
        input_data = {key.split(" ")[0]: float(entry.get()) for key, entry in entries.items()}

        # 获取单选框的值
        input_data.update({key: var.get() for key, var in radio_vars.items()})

        # 处理离散变量
        discrete_mapping = {
            "饮酒": "饮酒_2",
            "近视的度数": {2: "近视的度数_2", 3: "近视的度数_3", 4: "近视的度数_4"},
            "成年期白天的情绪": "成年期白天的情绪_2",
            "直系亲属是否有疾病史": "直系亲属是否有疾病史_2"
        }

        input_df = pd.DataFrame([input_data])
        
        for col, mapping in discrete_mapping.items():
            if isinstance(mapping, dict):  # 处理多个可能的独热编码
                for val, col_name in mapping.items():
                    input_df[col_name] = 1 if input_df[col].iloc[0] == val else 0
            else:  # 处理单个独热编码
                input_df[mapping] = 1 if input_df[col].iloc[0] == 2 else 0

        # 删除原始的离散列
        input_df.drop(columns=["饮酒", "近视的度数", "成年期白天的情绪", "直系亲属是否有疾病史"], inplace=True)

        # 补充缺失的列，并确保顺序一致
        missing_cols = set(expected_columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0  # 缺失的独热编码列补 0

        input_df = input_df[expected_columns]  # 确保列顺序一致

        # 标准化输入数据
        input_scaled = scaler.transform(input_df)

        # 进行预测
        prediction = bp_model.predict(input_scaled)

        # 显示预测结果
        if prediction[0] == 1:
            messagebox.showinfo("预测结果", "✅ 预测结果：属于该类别 (1)")
        else:
            messagebox.showwarning("预测结果", "⚠️ 预测结果：不属于该类别 (0)")

    except Exception as e:
        messagebox.showerror("错误", f"预测失败: {str(e)}")

# 预测按钮
predict_button = ttk.Button(scrollable_frame, text="🚀 进行预测", command=predict, bootstyle="success outline", padding=10)
predict_button.grid(row=len(input_fields) + len(radio_options) + 1, column=0, columnspan=2, pady=20)

# 运行 Tkinter 主循环
root.mainloop()