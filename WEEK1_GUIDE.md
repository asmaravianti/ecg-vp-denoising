# 第一周任务完成指南

## 已完成的任务

✅ **任务1**: 实现MIT-BIH数据加载器并进行窗口化处理
✅ **任务2**: 集成NSTDB噪声混合和SNR控制功能
✅ **任务3**: 确定WWPRD的权重并进行基本评估
✅ **任务4**: 创建训练脚本并生成训练损失曲线
✅ **任务5**: 创建评估脚本生成PRD/WWPRD图表

## 项目结构

```
ecg-vp-denoising/
├── ecgdae/
│   ├── data.py          # MIT-BIH和NSTDB数据加载器
│   ├── losses.py        # PRD, WWPRD损失函数
│   ├── models.py        # 卷积自编码器模型
│   └── metrics.py       # 评估指标（PRD, WWPRD, SNR等）
├── scripts/
│   ├── train_mitbih.py      # 训练脚本
│   └── evaluate_mitbih.py   # 评估脚本
├── requirements.txt     # Python依赖
└── WEEK1_GUIDE.md      # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包括:
- PyTorch (深度学习框架)
- wfdb (MIT-BIH数据加载)
- matplotlib, seaborn (可视化)
- numpy, scipy (数值计算)

### 2. 训练模型

使用WWPRD损失函数训练模型（推荐配置）:

```bash
python scripts/train_mitbih.py \
    --num_records 10 \
    --noise_type nstdb \
    --nstdb_noise muscle_artifact \
    --snr_db 10.0 \
    --loss_type wwprd \
    --weight_alpha 2.0 \
    --hidden_dims 32 64 128 \
    --latent_dim 32 \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-3 \
    --output_dir ./outputs/week1 \
    --save_model
```

**主要参数说明:**
- `--num_records`: 使用的MIT-BIH记录数量（快速训练用10，完整训练用全部48条）
- `--noise_type`: 噪声类型（`nstdb`真实ECG噪声或`gaussian`高斯噪声）
- `--nstdb_noise`: NSTDB噪声类型
  - `muscle_artifact`: 肌肉伪影（推荐）
  - `baseline_wander`: 基线漂移
  - `electrode_motion`: 电极运动伪影
- `--snr_db`: 信噪比（分贝），默认10dB
- `--loss_type`: 损失函数
  - `wwprd`: 加权PRD损失（**推荐用于诊断质量**）
  - `prd`: 标准PRD损失
  - `stft_wwprd`: 频域加权PRD
  - `mse`: 均方误差（基线对比）
- `--weight_alpha`: WWPRD权重系数（控制QRS复合波强调程度）
- `--latent_dim`: 潜在空间维度（控制压缩率）

### 3. 训练输出

训练完成后，`./outputs/week1/` 目录包含:

```
outputs/week1/
├── config.json                    # 训练配置
├── best_model.pth                 # 最佳模型检查点
├── training_history.json          # 训练历史数据
├── final_metrics.json             # 最终评估指标
├── training_curves.png            # 训练损失曲线图
└── reconstruction_examples.png    # 重建示例
```

**训练曲线图** (`training_curves.png`) 包含4个子图:
1. 训练和验证损失
2. PRD随训练变化（带质量阈值线）
3. WWPRD随训练变化（带质量阈值线）
4. SNR改善情况

### 4. 评估模型

训练完成后，使用评估脚本生成详细的PRD/WWPRD分析:

```bash
python scripts/evaluate_mitbih.py \
    --model_path ./outputs/week1/best_model.pth \
    --config_path ./outputs/week1/config.json \
    --output_dir ./outputs/week1/evaluation \
    --num_samples 500
```

### 5. 评估输出

评估脚本生成以下可视化和分析结果:

```
outputs/week1/evaluation/
├── evaluation_metrics.json        # 所有样本的指标数据
├── metric_distributions.png       # PRD/WWPRD/SNR分布图
├── prd_wwprd_scatter.png         # PRD vs WWPRD散点图
├── quality_classification.png     # 质量分类饼图
└── reconstruction_gallery.png     # 最佳/最差重建示例
```

**生成的图表:**

1. **metric_distributions.png** - 4个子图:
   - PRD分布直方图（带质量阈值）
   - WWPRD分布直方图（带质量阈值）
   - 输入/输出SNR对比
   - SNR改善分布

2. **prd_wwprd_scatter.png**:
   - PRD vs WWPRD散点图
   - 颜色表示SNR改善程度
   - 显示质量阈值线

3. **quality_classification.png**:
   - PRD质量分类饼图
   - WWPRD质量分类饼图
   - 显示"优秀/很好/好/不好"各占比例

4. **reconstruction_gallery.png**:
   - 展示8个重建示例（前4个最佳，后4个最差）
   - 每个示例显示：干净信号、噪声信号、重建信号
   - 标注PRD、WWPRD和SNR改善值

## 质量标准

### PRD (Percent Root-mean-square Difference)

| 质量等级 | PRD范围 | 诊断可用性 |
|---------|---------|-----------|
| **优秀** (Excellent) | < 4.33% | 完全可用于诊断 |
| **很好** (Very Good) | 4.33% - 9.00% | 可用于诊断 |
| **好** (Good) | 9.00% - 15.00% | 有限诊断价值 |
| **不好** (Not Good) | ≥ 15.00% | 不推荐诊断使用 |

### WWPRD (Waveform-Weighted PRD)

| 质量等级 | WWPRD范围 | 诊断可用性 |
|---------|----------|-----------|
| **优秀** (Excellent) | < 7.4% | 完全可用于诊断 |
| **很好** (Very Good) | 7.4% - 14.8% | 可用于诊断 |
| **好** (Good) | 14.8% - 24.7% | 有限诊断价值 |
| **不好** (Not Good) | ≥ 24.7% | 不推荐诊断使用 |

**说明**: WWPRD比PRD更重视QRS复合波区域（心室去极化），因此更能反映诊断信息的保留程度。

## WWPRD权重计算

WWPRD使用基于信号导数的权重，强调快速变化的区域（如QRS波）:

```python
# 权重计算公式
derivative = |gradient(signal)|
normalized_deriv = derivative / max(derivative)
weights = 1 + alpha * normalized_deriv
```

- `alpha`: 控制权重强调程度（默认2.0）
  - 较小的alpha（1.0）: 权重更均匀
  - 较大的alpha（3.0+）: 更强调QRS区域

## 实验配置建议

### 配置1: 快速原型（开发测试）
```bash
python scripts/train_mitbih.py \
    --num_records 5 \
    --epochs 20 \
    --batch_size 32 \
    --hidden_dims 16 32 64 \
    --latent_dim 16
```
- 训练时间: ~5-10分钟
- 用途: 代码测试和快速迭代

### 配置2: 标准训练（第一周交付）
```bash
python scripts/train_mitbih.py \
    --num_records 10 \
    --noise_type nstdb \
    --nstdb_noise muscle_artifact \
    --snr_db 10.0 \
    --loss_type wwprd \
    --weight_alpha 2.0 \
    --hidden_dims 32 64 128 \
    --latent_dim 32 \
    --epochs 50 \
    --save_model
```
- 训练时间: ~30-60分钟
- 用途: 第一周任务交付

### 配置3: 完整训练（最终报告）
```bash
python scripts/train_mitbih.py \
    --num_records 40 \
    --noise_type nstdb \
    --nstdb_noise muscle_artifact \
    --snr_db 10.0 \
    --loss_type wwprd \
    --weight_alpha 2.0 \
    --model_type residual \
    --hidden_dims 32 64 128 \
    --latent_dim 32 \
    --epochs 100 \
    --save_model
```
- 训练时间: ~2-4小时
- 用途: 最终性能评估

## 关键特性

### 1. MIT-BIH数据加载器 (`ecgdae/data.py`)
- ✅ 自动从PhysioNet下载MIT-BIH数据
- ✅ 支持窗口化处理（可配置窗口长度和步长）
- ✅ 支持多通道ECG（默认使用通道0 - MLII导联）
- ✅ 自动归一化（零均值，单位方差）

### 2. NSTDB噪声混合器 (`ecgdae/data.py`)
- ✅ 支持3种真实ECG噪声类型:
  - 肌肉伪影 (muscle_artifact)
  - 基线漂移 (baseline_wander)
  - 电极运动 (electrode_motion)
- ✅ 精确的SNR控制
- ✅ 可选高斯噪声混合
- ✅ 噪声缓存机制（加速训练）

### 3. 可微分WWPRD损失 (`ecgdae/losses.py`)
- ✅ PRD损失
- ✅ 时域WWPRD损失（基于导数权重）
- ✅ 频域WWPRD损失（基于STFT）
- ✅ 所有损失函数完全可微分，支持梯度优化

### 4. 自编码器模型 (`ecgdae/models.py`)
- ✅ 标准卷积自编码器
- ✅ 残差自编码器（更好的梯度流）
- ✅ 可配置压缩率（通过latent_dim控制）
- ✅ BatchNorm和GELU激活函数

### 5. 评估指标 (`ecgdae/metrics.py`)
- ✅ PRD（百分比均方根差）
- ✅ WWPRD（波形加权PRD）
- ✅ SNR（信噪比）
- ✅ 压缩率计算
- ✅ 自动质量分类

## 下一步工作（第2-4周）

### 第2周: 压缩率分析
- [ ] 实现潜在空间量化（4-8位）
- [ ] 计算实际压缩率
- [ ] 生成PRD-CR和WWPRD-CR曲线
- [ ] 对比不同压缩率下的性能

### 第3周: 损失函数对比和VP层
- [ ] 损失函数对比（MSE vs PRD vs WWPRD）
- [ ] 实现变量投影（VP）层
- [ ] VP层与标准卷积层对比

### 第4周: 最终报告
- [ ] 完整数据集训练
- [ ] 最佳配置确定
- [ ] 准备报告和演示幻灯片

## 故障排除

### 问题1: 无法下载MIT-BIH数据
**解决方案**: wfdb库会自动从PhysioNet下载，如果网络有问题:
```python
# 手动下载到指定目录
import wfdb
wfdb.dl_database('mitdb', dl_dir='./data/mitbih')
```

### 问题2: CUDA内存不足
**解决方案**: 减小batch_size或模型大小:
```bash
--batch_size 16 --hidden_dims 16 32 64 --latent_dim 16
```

### 问题3: 训练速度慢
**解决方案**:
- 减少记录数量: `--num_records 5`
- 使用GPU: 自动检测CUDA
- 减少epochs: `--epochs 20`

## 联系和反馈

如有问题或建议，请查看代码注释或联系项目维护者。

---

**祝训练顺利！** 🚀

