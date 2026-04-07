from modelscope.msdatasets import MsDataset
import os

# 指定你的目标路径
target_dir = '/data/siyu.liu/datasets/'

# 确保文件夹存在（如果不存在则创建）
os.makedirs(target_dir, exist_ok=True)

# 加载数据集并指定缓存目录
ds = MsDataset.load(
    'DDLteam/MFFI',
    cache_dir=target_dir  # 这里指定下载位置
)

print(f"数据集已下载/加载到: {target_dir}")