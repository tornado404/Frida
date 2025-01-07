import time

from pymycobot import MyCobotSocket
# Port 9000 is used by default
mc = MyCobotSocket("192.168.31.8", 9000)
mc.power_on()
print("is power on:", mc.is_power_on())
print("is_controller_connected:", mc.is_controller_connected())

res = mc.get_angles()
print(res)

time.sleep(1)  # 等2秒
mc.set_color(255, 0, 0)  # 红灯亮
mc.sync_send_angles([0, 0, 0, 0, 0, -45], 20)
# # 打开夹爪
# mc.set_gripper_state(0, 70)
# time.sleep(2)  # 等2秒
# 关闭自由移动模式
# mc.set_free_mode(0)
# mc.sync_send_angles([0, 0, 0, 0, 0, -45], 20)
# # 初始化完成, 绿灯亮
# mc.set_color(0, 255, 0)
#
mc.send_angles([-45,0,0,0,0,0],40)
print(res)