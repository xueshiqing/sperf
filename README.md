# sperf
基于异构负载特征预测的资源调度系统——数据处理

### 数据处理部分

test.py: 生成模型
meta_collect.py：训练数据采集
timeMerge.py：多维度数据按时间戳汇聚
merge.py：多轮次数据拼接

### 测试脚本部分
get_output.py：获取实时接口延迟输出（系统演示）
normal_submit：NMP情况测试自动化脚本
olc_submit：OLC情况测试自动化脚本
forecast_submit：MP情况测试自动化脚本