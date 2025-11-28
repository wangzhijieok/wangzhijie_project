import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 导入处理图标
df = pd.read_excel('数据1.xlsx', header=0, skiprows=[1, 2])
print(df.head())

height_col = 'height'
weight_col = 'weight'

plt.figure(figsize=(6, 4))
sns.scatterplot(x=height_col, y=weight_col, data=df, color='royalblue')
plt.title('身高与体重关系散点图')
plt.xlabel('身高 (cm)')
plt.ylabel('体重 (kg)')
plt.grid(alpha=0.3)
plt.show()

corr = df[height_col].corr(df[weight_col])
print(f"身高 体重的皮尔逊相关系数", corr)

# 回归模型
x = sm.add_constant(df[height_col])
y = df[weight_col]

model = sm.OLS(y, x).fit()
print(model.summary())