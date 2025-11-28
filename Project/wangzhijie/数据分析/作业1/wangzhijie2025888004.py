import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False


df = pd.read_excel('平安银行 (1).xlsx')

print("原始数据预览：")
print(df.head())
print("\n数据概况：")
print(df.info())

print(df.isnull().sum())