from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import numpy as np

# 定义机械臂的DH参数
my_chain = Chain(name='6-axis Arm', links=[
    OriginLink(),
    URDFLink(
        name="joint1",
        origin_translation=[0, 0, 0.13156],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 0]
    ),
    URDFLink(
        name="joint2",
        origin_translation=[0.1104, 0, 0],
        origin_orientation=[-np.pi/2, 0, 0],
        rotation=[0, 0, 0]
    ),
    URDFLink(
        name="joint3",
        origin_translation=[0.096, 0, 0],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 0]
    ),
    URDFLink(
        name="joint4",
        origin_translation=[0, 0, 0.06462],
        origin_orientation=[-np.pi/2, 0, 0],
        rotation=[0, 0, 0]
    ),
    URDFLink(
        name="joint5",
        origin_translation=[0, 0, 0.07318],
        origin_orientation=[np.pi/2, 0, 0],
        rotation=[0, 0, 0]
    ),
    URDFLink(
        name="joint6",
        origin_translation=[0, 0, 0.0486],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 0]
    )
])

def calculate_joint_angles(target_position):
    joint_angles = my_chain.inverse_kinematics(target_position)
    return joint_angles

def test_calculate_joint_angles():
    # 测试目标位置
    target_position = [0.120, -0.208, 0.087]

    # 计算关节角度
    joint_angles = calculate_joint_angles(target_position)
    print(joint_angles)

 
if __name__ == "__main__":
    test_calculate_joint_angles()

