milCode/                  # 你的项目主目录
│
├── data/                        # 📁 [原始数据] 存放所有患者的原始内窥镜/病理图片
│   ├── 1/                       # 文件夹名为 patient_id (如 "1")
│   │   ├── image_01.jpg
│   │   └── image_02.jpg
│   ├── 2/                       # 文件夹名为 patient_id (如 "2")
│   │   └── ...
│   └── ...
│
├── weights/                     # 📁 [预训练权重] 存放下载好的大模型离线权重
│   ├── dinov2_vitb14_pretrain.pth
│   └── efficientnet_b0_rwightman-3dd342df.pth
│
├── labels.xlsx                  # 📊 [全局标签] 包含患者ID(id)和各种疾病列的Excel表
│
├── main.ipynb                   # 🌟 [全局控制枢纽] 你的 Jupyter Notebook，串联以下所有脚本
│
├── extract_instance_features.py # ⚙️ [脚本 1] 负责读取 data/ 下的图片，提特征并导出 Excel
├── dataset_patient.py           # ⚙️ [脚本 2] 包含两个 Dataset 类：一个用于读取原图提特征，一个用于读取 .pt 训练
├── mil_train.py                 # ⚙️ [脚本 3] 存放你的神经网络架构 (Encoder, AttentionPool, Classifier)
├── train_log.py                 # ⚙️ [脚本 4] 核心训练引擎，包含混合损失函数、优化器和 5 折交叉验证逻辑
│
├── outputs/                     # 📂 [自动生成的输出目录]
│   ├── encoder_instance_features.xlsx  # (生成物) 第一步跑完后生成的巨大特征表
│   ├── train_20260303_xxxx.log         # (生成物) 第三步跑完后的训练日志
│   └── best_fold1.ckpt                 # (生成物) 训练保存的最佳模型权重
│
└── encoder_features/            # 📂 [自动生成的特征张量目录]
    ├── 1.pt                     # (生成物) 第二步拆分出来的患者 1 的特征张量
    ├── 2.pt                     # (生成物) 第二步拆分出来的患者 2 的特征张量


服务器开启jupyter端口
jupyter lab --no-browser --port=8888

本地连接
ssh -L 8000:localhost:8888 zhh@10.101.2.13

此文件把是对胡老师的milCode项目的说明
完全照搬胡老师的数据


