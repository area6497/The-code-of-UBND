# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 21:06:05 2025

@author: user
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 混淆矩阵数据
cm = np.array([[516, 9],
               [27, 225]])

labels = ['benign', 'malignant']

# 绘制混淆矩阵
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('')
plt.savefig('confusion_matrix.jpg', dpi=300, bbox_inches='tight')
plt.show()
