import numpy as np
import plotly.graph_objects as go

# 假设你有一个3D numpy数组'sdf'，它包含了你的SDF值
sdf = np.random.rand(10, 10, 10)  # 用随机值作为示例

# 获取sdf数组中的每个元素的x，y，z坐标
x, y, z = np.indices(sdf.shape)

# 创建一个3D散点图，其中颜色映射到sdf值
fig = go.Figure(data=go.Scatter3d(
    x=x.flatten(),
    y=y.flatten(),
    z=z.flatten(),
    mode='markers',
    marker=dict(
        size=2,
        color=sdf.flatten(),  # 设置颜色为sdf值
        colorscale='Viridis',  # 选择颜色范围
        opacity=0.8
    )
))

# 保存图像到HTML文件
fig.write_html('sdf_visualization.html')
