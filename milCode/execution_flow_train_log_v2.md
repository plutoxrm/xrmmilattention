# train\_log\_v2.py 执行流程说明

这版脚本解决了两个核心问题：

1. 在原有 AUROC / AUPRC / Sens@95Spec / Brier 的基础上，新增了 ACC、F1、Precision、Recall、Specificity。
2. 训练阶段允许固定数量实例采样；验证阶段使用患者全部实例进行 full-bag 推断。
3. 阈值不是在验证集上挑，而是在每个训练 fold 内部确定，然后用于该 fold 的验证集。

\---

## 一、整体思想

整个流程分成三条线：

* **训练线**：训练集按固定实例数采样，用于反向传播。
* **训练评估线**：训练集 full-bag 推断，用于阈值选择。
* **验证评估线**：验证集 full-bag 推断，用于最终验证指标。

因此，每个 fold 中会构建 3 个 dataset：

1. `train\_ds`：真正训练，使用 `--train\_max\_feats` 与 `--train\_instance\_strategy`
2. `train\_eval\_ds`：训练阈值选择，使用 `--valid\_max\_feats` 与 `--valid\_instance\_strategy`
3. `valid\_ds`：验证评估，使用 `--valid\_max\_feats` 与 `--valid\_instance\_strategy`

\---

## 二、脚本启动后做什么

### 1\. 读取命令行参数

关键参数：

* `--feat\_dir`：患者级 pt 特征目录
* `--labels\_csv`：标签表
* `--label\_cols`：目标标签列
* `--train\_max\_feats`：训练阶段每个患者最多使用多少个实例
* `--train\_instance\_strategy`：训练阶段实例抽样方式
* `--valid\_max\_feats`：验证阶段最大实例数，通常设为 `-1`
* `--valid\_instance\_strategy`：验证阶段实例策略，通常设为 `all`
* `--threshold\_metric`：训练 fold 内阈值选择准则，可选 `youden` 或 `f1`

\---

### 2\. 建立日志

脚本会把所有 `print` 同时输出到屏幕和日志文件：

* `outputs/train\_YYYYMMDD\_HHMMSS.log`

\---

### 3\. 固定随机种子

固定 Python、NumPy、PyTorch 的随机种子，减少实验波动。

\---

### 4\. 读取标签表

支持 Excel / CSV，并且会把 `id` 统一转成字符串，避免和 `encoder\_features/<pid>.pt` 文件名不一致。

\---

### 5\. 自动推断特征维度

脚本会去 `feat\_dir` 中找到一个可用的 `.pt` 文件，读取其中：

* `data\['feats'].shape\[1]`

作为 `in\_dim`。

这样就不用再手动写死 `768`。

\---

## 三、交叉验证阶段做什么

### 1\. 自动选择折分方式

* 单标签任务：`StratifiedKFold`
* 多标签任务：`MultilabelStratifiedKFold`

你现在的 `代谢慢病` 是单标签任务，因此会用 `StratifiedKFold`。

\---

### 2\. 每个 fold 构建 3 套数据

#### （1）训练数据 `train\_ds`

使用：

* `--train\_max\_feats`
* `--train\_instance\_strategy`

例如：

* `train\_max\_feats=30`
* `train\_instance\_strategy=random`

表示每轮训练时，每个患者最多随机抽 30 个实例。

这是**训练近似策略**，目的是降低计算成本。

#### （2）训练阈值选择数据 `train\_eval\_ds`

使用：

* `--valid\_max\_feats`
* `--valid\_instance\_strategy`

通常设置为：

* `valid\_max\_feats=-1`
* `valid\_instance\_strategy=all`

这表示：**训练 fold 内部做 full-bag 推断，用全部实例选阈值**。

#### （3）验证数据 `valid\_ds`

和 `train\_eval\_ds` 一样，通常也是 full-bag。

这表示：**验证阶段用该患者全部实例推断，得到更真实的患者级性能**。

\---

## 四、为什么需要两个不同的 collate

### 1\. 训练 collate：`collate\_train\_fixed`

训练阶段由于所有患者都被裁剪/补齐到同样长度，所以可以直接：

* `torch.stack(feats)`

因此训练 DataLoader 可以使用普通 batch，例如 `batch\_size=8`。

### 2\. 验证 collate：`collate\_eval\_variable`

验证阶段使用 full-bag，每个患者实例数不同，不能直接 stack 到同一张量中。

解决办法：

* 验证 DataLoader 强制 `batch\_size=1`
* 每次只处理一个患者
* 把这个患者的 `\[N, D]` 扩成 `\[1, N, D]`

这样模型仍然兼容，不需要改 `PatientMILFeatures`。

\---

## 五、每个 epoch 内部执行顺序

每个 epoch 的完整流程如下：

### 第 1 步：训练

使用 `train\_ds`：

* 每个患者按训练策略采样
* 前向计算
* 计算 loss
* 反向传播
* 更新参数

loss 支持两种：

* `BCEWithLogitsLoss`
* `CombinedLoss = BCE + AUCMLoss`

\---

### 第 2 步：训练 fold 内 full-bag 推断

训练完一轮后，不立刻在验证集上找阈值。

而是先用 `train\_eval\_ds` 对训练 fold 全体患者做 full-bag 推断，得到：

* `train\_scores`
* `train\_labels`

\---

### 第 3 步：在训练 fold 内确定阈值

使用 `find\_best\_threshold\_binary()` 在训练 fold 上找阈值。

支持两种准则：

#### `youden`

最大化：

* `sensitivity + specificity - 1`

#### `f1`

最大化：

* `F1 score`

这样做的目的是：

* 不在验证集上调阈值
* 避免验证信息泄漏
* 使 ACC / F1 等阈值相关指标更规范

\---

### 第 4 步：验证集 full-bag 推断

再用 `valid\_ds` 对验证集每个患者做 full-bag 推断，得到：

* `valid\_scores`
* `valid\_labels`

然后使用第 3 步从训练 fold 得到的阈值，去计算验证指标。

\---

### 第 5 步：计算验证指标

使用 `compute\_binary\_metrics()` 计算：

* AUROC
* AUPRC
* ACC
* F1
* Precision
* Recall
* Specificity
* Sens@95Spec
* Brier
* Threshold

其中：

* AUROC / AUPRC / Sens@95Spec / Brier 是概率层面指标
* ACC / F1 / Precision / Recall / Specificity 是阈值后的分类指标

\---

### 第 6 步：选择 best epoch

脚本仍然使用：

* **验证集 AUROC 最大**

作为 best epoch 标准。

这样可以保持主指标稳定，不会因为阈值类指标波动导致 best model 选择混乱。

如果当前 epoch 是 best epoch：

* 更新 `best\_metrics`
* 可选保存 checkpoint

\---

## 六、保存了哪些文件

### 1\. 日志文件

* `outputs/train\_时间戳.log`

### 2\. 每折最优模型（可选）

只有加了 `--save\_ckpt` 才会保存：

* `outputs/best\_fold1.pt`
* `outputs/best\_fold2.pt`
* ...

其中包含：

* 模型参数
* 优化器参数
* 最优 epoch
* 阈值
* 最优指标
* 当前运行参数

### 3\. 交叉验证汇总表

* `outputs/cv\_summary\_时间戳.csv`

包含每一折的：

* fold
* best\_epoch
* auc
* auprc
* acc
* f1
* precision
* recall
* specificity
* sens95
* brier
* threshold

\---

## 七、推荐运行方式

推荐命令：

```python
%run train\_log\_v2.py \\
  --feat\_dir ./encoder\_features \\
  --labels\_csv ./labels03.xlsx \\
  --label\_cols 代谢慢病 \\
  --epochs 100 \\
  --batch\_size 8 \\
  --lr 5e-4 \\
  --weight\_decay 1e-4 \\
  --train\_max\_feats 30 \\
  --train\_instance\_strategy random \\
  --valid\_max\_feats -1 \\
  --valid\_instance\_strategy all \\
  --folds 5 \\
  --num\_workers 0 \\
  --architecture attention \\
  --use\_combined\_loss \\
  --auc\_weight 0.5 \\
  --threshold\_metric youden \\
  --save\_ckpt \\
  --seed 42
```

\---

## 八、论文里可以怎么写

你现在可以更规范地描述方法：

> 为控制训练成本，训练阶段对每个患者随机采样固定数量实例；
> 为真实评估患者级诊断性能，阈值在训练 fold 内确定，验证阶段使用患者全部可用实例进行推断。

这样比“训练和验证都固定抽样若干实例”更有说服力，也更不容易被审稿人质疑。

