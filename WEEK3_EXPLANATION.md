# Week 3 详细解释

## Week 3 要做什么？

Week 3 主要有**4个任务**：

### 1. **Loss Ablation（损失函数对比实验）**
### 2. **Noise Ablation（噪声对比实验）**
### 3. **Bottleneck Sweep（压缩比扫描）**
### 4. **VP Layer（可变投影层）**

---

## 什么是 Ablation Study（消融实验）？

**Ablation Study** = 对比实验，用来理解每个组件的作用

**例子**：
- 想知道"用MSE训练"和"用WWPRD训练"哪个更好？
- 想知道"加噪声训练"和"不加噪声训练"有什么区别？
- 这些就是 **Ablation Study**

---

## 任务1: Loss Ablation（损失函数对比）

### 目的
比较不同的损失函数，看哪个效果最好

### 实验
训练**3个相同的模型**，但用**不同的损失函数**：
1. **MSE** (Mean Squared Error) - 传统的均方误差
2. **PRD** (Percent Root-mean-square Difference) - PRD损失
3. **WWPRD** (Waveform-Weighted PRD) - 加权PRD损失

### 为什么做这个？
- 验证WWPRD损失是否真的比MSE好
- 看看PRD损失和WWPRD损失有什么区别
- 为论文提供实验证据

### 结果
会得到对比表格：
| 损失函数 | PRD | WWPRD | SNR改进 |
|---------|-----|-------|---------|
| MSE | ?% | ?% | ? dB |
| PRD | ?% | ?% | ? dB |
| WWPRD | ?% | ?% | ? dB |

---

## 任务2: Noise Ablation（噪声对比实验）

### 目的
比较"加噪声训练"和"不加噪声训练"的区别

### 实验
训练**2个相同的模型**：
1. **With Noise**: 输入信号加噪声（SNR=10dB），训练去噪
2. **Without Noise**: 输入信号不加噪声，直接训练压缩

### 为什么做这个？
- 验证噪声训练是否真的能提高去噪能力
- 看看模型是否真的学会了去噪

### 结果
会看到：
- 加噪声训练的模型：去噪效果好，但压缩可能稍差
- 不加噪声的模型：压缩可能更好，但去噪能力差

---

## 任务3: Bottleneck Sweep（压缩比扫描）

### 目的
验证**Rate-Distortion（率失真）曲线**是否单调

### 什么是Rate-Distortion？
- **Rate** = 压缩比（Compression Ratio, CR）
- **Distortion** = 失真（PRD/WWPRD）
- **关系**：压缩比越高 → 失真越大（这是正常的）

### 实验
训练**多个模型**，每个模型用**不同的latent_dim**：
- latent_dim = 8 → 高压缩比，高失真
- latent_dim = 16 → 中等压缩比，中等失真
- latent_dim = 32 → 低压缩比，低失真
- ... 等等

### 为什么做这个？
- 验证理论：压缩比越高，质量越差（单调性）
- 找到最佳平衡点（压缩比 vs 质量）

### 结果
会得到一条曲线：
```
PRD vs Compression Ratio
    |
高  |     ●
    |   ●
PRD | ●
    |●
低  |________________
    低CR    高CR
```

---

## 任务4: VP Layer（可变投影层）- 重点！

### 什么是VP Layer？

**VP = Variable Projection（可变投影）**

这是一个**特殊的神经网络层**，用来替代标准的卷积层。

### 为什么需要VP Layer？

**当前问题**：
- 标准卷积层：固定的压缩方式
- 如果要改变压缩比，需要重新训练整个模型

**VP Layer的优势**：
- **自适应压缩**：可以根据信号特性自动调整
- **更灵活**：一个模型可以适应不同的压缩比
- **可能更好的性能**：理论上可以更好地保留重要信息

### VP Layer vs 标准卷积

**标准卷积（Conv）**：
```
输入信号 → [固定卷积核] → 输出特征
```

**VP Layer**：
```
输入信号 → [自适应投影矩阵] → 输出特征
         ↑
    根据信号特性调整
```

### 具体实现

VP Layer会替换encoder的**第一个卷积层**：
```
原来的模型：
Input → Conv1 → Conv2 → Conv3 → Bottleneck

VP模型：
Input → VP Layer → Conv2 → Conv3 → Bottleneck
```

### 为什么放在第一个位置？
- 第一个层处理原始信号，影响最大
- 可以最早捕获信号的重要特征
- 实验证明这个位置最有效

### VP Layer的数学原理（简化）

标准卷积：
```
y = Conv(x, W)  # W是固定的卷积核
```

VP Layer：
```
y = Project(x, W_adaptive)  # W_adaptive根据x自适应调整
```

具体来说，VP Layer使用**有理函数投影**（Rational Function Projection）：
- 不是固定的线性变换
- 而是根据输入信号自适应调整的投影
- 可以更好地保留信号的重要特征

### 参考论文
项目中有PDF文件：
`Generalized_Rational_Variable_Projection_With_Application_in_ECG_Compression(1).pdf`

这个论文详细解释了VP方法在ECG压缩中的应用。

---

## Week 3 的完整流程

### 步骤1: Loss Ablation（6个实验）
```
训练6个模型：
- CR≈8: MSE, PRD, WWPRD (3个)
- CR≈16: MSE, PRD, WWPRD (3个)
```

### 步骤2: Noise Ablation（2个实验）
```
训练2个模型：
- 加噪声训练
- 不加噪声训练
```

### 步骤3: Bottleneck Sweep（6-8个实验）
```
训练多个模型：
- latent_dim = 8, 12, 16, 20, 24, 32, 40, 48
- 生成Rate-Distortion曲线
```

### 步骤4: VP Layer（1-2个实验）
```
1. 实现VP Layer代码
2. 训练VP模型
3. 与标准卷积模型对比
```

---

## 为什么Week 3很重要？

### 1. **科学严谨性**
- Ablation study证明每个组件的必要性
- 为论文提供实验证据

### 2. **理解模型行为**
- 知道哪个损失函数最好
- 知道噪声训练是否必要
- 知道压缩比和质量的关系

### 3. **创新点**
- VP Layer是项目的创新点之一
- 如果VP Layer效果好，是很好的贡献

---

## 时间安排

| 任务 | 实验数 | 每个时间 | 总时间 |
|------|--------|----------|--------|
| Loss ablation | 6 | 10-16h | 60-96h |
| Noise ablation | 2 | 10-16h | 20-32h |
| Bottleneck sweep | 6-8 | 10-16h | 60-128h |
| VP Layer | 1-2 | 10-16h + 实现时间 | ~20h |

**总计**: 约1-2周（可以并行运行一些实验）

---

## 总结

**Week 3 = 对比实验 + 新方法**

1. **Loss Ablation**: 证明WWPRD比MSE好
2. **Noise Ablation**: 证明噪声训练有效
3. **Bottleneck Sweep**: 验证理论，找最佳平衡点
4. **VP Layer**: 实现新方法，可能提升性能

**VP Layer** = 一个可以替代卷积层的自适应投影层，可能让模型更灵活、性能更好。

---

## 下一步

1. **现在**: 可以开始Loss Ablation的测试（30 epochs，2-3小时）
2. **主训练完成后**: 开始完整实验
3. **VP Layer**: 需要先理解论文，然后实现代码

需要我帮你开始哪个实验？或者你想先了解更多VP Layer的细节？

