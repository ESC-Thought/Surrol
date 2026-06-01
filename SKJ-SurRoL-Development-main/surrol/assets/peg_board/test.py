import pybullet as p
import pybullet_data

# 初始化 PyBullet 仿真环境
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 加载 URDF 文件
peg_board_id = p.loadURDF("./peg_board.urdf", basePosition=[0, 0, 0])

# 获取 URDF 中的所有链接
num_joints = p.getNumJoints(peg_board_id)

# 遍历所有链接，查找名字以 "peg" 开头的链接
peg_positions = {}
for joint_index in range(num_joints):
    joint_info = p.getJointInfo(peg_board_id, joint_index)
    link_name = joint_info[12].decode("utf-8")  # 获取链接名字
    
    if "peg" in link_name:  # 筛选名字包含 "peg" 的链接
        link_state = p.getLinkState(peg_board_id, joint_index)
        link_position = link_state[0]  # 获取链接的世界坐标位置
        peg_positions[link_name] = link_position

# 输出所有 peg 的 ID 和位置
for peg_name, position in peg_positions.items():
    print(f"Peg Name: {peg_name}, Position: {position}")

# 关闭 PyBullet
p.disconnect()
