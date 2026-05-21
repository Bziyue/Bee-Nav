# Bee-Nav Blender 轻量仿真后端使用说明

本文档说明如何在 Windows + Conda/Python 环境中，使用本机 Blender 4.2 替代 Isaac Sim 完成 Bee-Nav 的轻量视觉仿真流程。

## 1. 这个后端做了什么

原始项目用 Isaac Sim 做三维场景渲染和相机拍图。这里新增的 Blender 后端保留原项目的训练数据格式：

```text
dataset/
  dataset_navigation.csv
  Replicator/rgb/rgb_0000.png
  Replicator/rgb/rgb_0001.png
  ...
```

流程为：

```text
Blender 搭轻量场景并拍图
-> 生成 dataset_navigation.csv
-> 用原项目 PyTorch CNN 训练
-> 仿真返航时每一步调用 Blender 拍当前视角
-> 网络预测回家方向
-> Python 更新位置并保存轨迹图
```

当前 smoke 示例使用 north-fixed 模式：相机始终朝北，网络学习世界坐标系下的回家向量。这样可以先稳定验证“拍图、训练、闭环返航”全流程。

## 2. 已确认的本机路径

Blender：

```powershell
C:\Program Files\Blender Foundation\Blender 4.2\blender.exe
```

Python/Conda 环境：

```powershell
C:\IL\env\python.exe
```

项目目录：

```powershell
C:\D\forcodex\Bee-Nav\visual_simulator_docker\CodeInsectInspired
```

## 3. 一键运行完整轻量仿真

在 PowerShell 中运行：

```powershell
Set-Location C:\D\forcodex\Bee-Nav\visual_simulator_docker\CodeInsectInspired
C:\IL\env\python.exe .\run_blender_smoke.py --output-dir windows_blender_outputs_final --n-positions 96 --image-size 128 --epochs 12
```

如果要使用 Kenney Nature Kit 小树林资产运行视觉返航流程：

```powershell
Set-Location C:\D\forcodex\Bee-Nav\visual_simulator_docker\CodeInsectInspired
C:\IL\env\python.exe .\run_blender_smoke.py --kenney-forest --output-dir windows_blender_kenney_run --n-positions 160 --image-size 128 --epochs 12 --flight-radius 7.5 --homing-max-steps 18
```

参数含义：

```text
--output-dir      输出目录
--n-positions     训练数据采样点数量，也就是 Blender 拍多少张训练图
--image-size      图片宽高，当前为 128x128
--epochs          CNN 训练轮数
--kenney-forest   使用随仓库提交的 Kenney CC0 小树林模型资产
```

## 4. 运行后输出

运行上面的命令后，会生成输出目录：

```powershell
C:\D\forcodex\Bee-Nav\visual_simulator_docker\CodeInsectInspired\windows_blender_outputs_final
```

关键输出：

```text
dataset/Replicator/rgb/       训练图片，共 96 张
dataset/dataset_navigation.csv 训练标签 CSV
homing_frames/                返航过程中每一步拍到的当前图像
homing_trajectories.png       返航轨迹图
summary.json                  运行摘要
```

此前在本机验证通过时的摘要为：

```json
{
  "backend": "blender",
  "success_count": 2,
  "num_runs": 2,
  "mean_angular_error": 20.436949926326207
}
```

也就是 2 条测试返航轨迹都进入了 home 阈值范围。当前仓库已清理生成结果；重新运行第 3 节命令即可再次生成这些图片、模型和摘要。

Kenney 小树林模式在本机验证时，160 张训练图、12 轮训练、3 条返航轨迹中有 2 条进入 home 阈值范围。它是轻量视觉闭环验证，不是论文级完整复现。

## 5. 查看可视化结果

运行第 3 节命令后，可以查看轨迹图：

```powershell
C:\D\forcodex\Bee-Nav\visual_simulator_docker\CodeInsectInspired\windows_blender_outputs_final\homing_trajectories.png
```

返航过程中拍到的图像：

```powershell
C:\D\forcodex\Bee-Nav\visual_simulator_docker\CodeInsectInspired\windows_blender_outputs_final\homing_frames
```

训练数据图片：

```powershell
C:\D\forcodex\Bee-Nav\visual_simulator_docker\CodeInsectInspired\windows_blender_outputs_final\dataset\Replicator\rgb
```

生成返航视频：

```powershell
C:\IL\env\python.exe .\make_homing_video.py --run-dir .\windows_blender_kenney_run --run-index 0
```

视频会输出到：

```text
windows_blender_kenney_run/kenney_homing_run_00.mp4
```

## 6. 新增和修改的文件

新增：

```text
blender_render_worker.py      Blender 内部执行的渲染脚本
blender_backend.py            普通 Python 调用 Blender 的后端封装
run_blender_smoke.py          一键完整流程：渲染、训练、返航、保存结果
make_homing_video.py          把返航帧和轨迹合成为 MP4 视频
render_kenney_forest_video.py 仅用于渲染 Kenney 小树林展示视频
BLENDER_BACKEND_使用说明.md    本说明文档
kenney_forest_assets/...      Kenney Nature Kit 的最小 CC0 模型子集
```

修改：

```text
data_loader.py
```

修改点：`num_workers` 现在可通过训练配置指定。Windows 上 smoke 流程使用 `num_workers=0`，避免 DataLoader 子进程递归问题。

## 7. 和 Isaac Sim 版本的区别

Blender 后端目前是轻量验证版：

```text
不模拟真实无人机动力学
不使用 Isaac Replicator
不使用 RTX 传感器
不导入复杂 USD 场景
```

它完成的是 Bee-Nav 这个项目真正需要的核心闭环：

```text
从三维环境拍当前视觉图像
-> 网络根据图像预测回家方向
-> Python 更新位置
-> 重复直到回家
```

后续如果要更接近原论文，可以继续扩展：

```text
1. 使用 Blender 全景相机替代当前透视相机
2. 支持导入真实 USD/OBJ/FBX 场景
3. 增加随机纹理、光照、地标形状
4. 改成常驻 Blender 服务，避免返航每一步启动一次 Blender
5. 恢复随机 yaw + 360 图像增强
```

当前版本的目标是：绕过本机 Isaac Sim 5.1 RTX 相机崩溃问题，先把 Bee-Nav 的完整视觉导航流程跑通并可视化。

## 8. 是否包含路径积分

当前 Blender/Kenney 流程跑的是视觉神经网络返航闭环：

```text
Blender 拍图 -> 几何真值生成 home 向量标签 -> CNN 训练 -> 当前视角预测 home 向量 -> 更新位置
```

这里没有接入论文中的路径积分噪声模型。论文完整流程中，路径积分主要用于学习飞行阶段生成监督标签；本轻量版本为了先验证 Windows 上的渲染、训练、返航和视频输出，直接用仿真真值位置生成标签。
