# -*- coding: utf-8 -*-
import time
import os
# export MESA_LOADER_DRIVER_OVERRIDE=swrast
os.environ["LIBGL_ALWAYS_INDIRECT"] = "0"
os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "swrast"
os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
import glfw
import mujoco
import numpy as np

# xvfb-run -a python /root/miniconda3/envs/mujoco-env/bin/python /mnt/d/code/frida/tests/test_mujoco.py
# 定义鼠标滚轮滚动的回调函数，用于处理界面缩放
def scroll_callback(window, xoffset, yoffset):
    # 使用 global 关键字声明 cam 为全局变量，以便在函数内部修改它
    global cam
    # 根据鼠标滚轮的垂直滚动量 yoffset 调整相机的距离，实现缩放效果
    # 0.1 是缩放的比例因子，可以根据需要调整
    cam.distance *= 1 - 0.1 * yoffset

# simulate /mnt/d/code/frida/MujocoBin/mujoco-3.3.2/model/humanoid/humanoid.xml
# m = mujoco.MjModel.from_xml_path(r'D:\code\frida\ROS\URDF\universal_robots_ur5e\scene.xml')
# m = mujoco.MjModel.from_xml_path(r'/mnt/d/code/frida/ROS/URDF/universal_robots_ur5e/scene.xml')

def main():
    # 声明 cam 为全局变量，方便在其他函数中使用
    global cam
    # 从指定的 XML 文件路径加载 MuJoCo 模型
    model = mujoco.MjModel.from_xml_path(r'/mnt/d/code/frida/ROS/URDF/universal_robots_ur5e/scene.xml')
    # 创建与模型对应的 MjData 实例，用于存储模拟过程中的动态数据
    data = mujoco.MjData(model)

    # 初始化 GLFW 库，用于创建窗口和处理输入事件
    if not glfw.init():
        # 如果初始化失败，直接返回
        return

    # 创建一个 1200x900 像素的窗口，标题为 'Panda Arm Control'
    window = glfw.create_window(1200, 900, 'Panda Arm Control', None, None)
    if not window:
        # 如果窗口创建失败，终止 GLFW 并返回
        print("窗口创建失败，终止 GLFW 并返回")
        glfw.terminate()
        return

    # 将当前上下文设置为新创建的窗口，以便后续的 OpenGL 操作在该窗口上进行
    glfw.make_context_current(window)

    # 设置鼠标滚轮事件的回调函数为 scroll_callback，当鼠标滚轮滚动时会调用该函数
    # glfw.set_scroll_callback(window, scroll_callback)

    # 初始化相机对象，用于定义观察视角
    cam = mujoco.MjvCamera()
    # 初始化渲染选项对象，用于设置渲染的一些参数
    opt = mujoco.MjvOption()
    # 设置相机的默认参数
    mujoco.mjv_defaultCamera(cam)
    # 设置渲染选项的默认参数
    mujoco.mjv_defaultOption(opt)
    # 初始化扰动对象，用于处理用户对模型的交互操作
    pert = mujoco.MjvPerturb()
    # 初始化渲染上下文对象，用于管理渲染资源
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    # 创建一个场景对象，用于存储要渲染的几何元素
    scene = mujoco.MjvScene(model, maxgeom=10000)

    # 根据名称 'hand' 查找末端执行器的 body ID
    end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
    print(f"End effector ID: {end_effector_id}")
    if end_effector_id == -1:
        # 如果未找到指定名称的末端执行器，打印警告信息并终止 GLFW
        print("Warning: Could not find the end effector with the given name.")
        glfw.terminate()
        return

    # 进入主循环，直到用户关闭窗口
    while not glfw.window_should_close(window):
        # 获取末端执行器在世界坐标系下的位置
        end_effector_pos = data.body(end_effector_id).xpos
        # 打印末端执行器的位置信息，方便调试
        print(f"End effector position: {end_effector_pos}")

        # 执行一步模拟，更新模型的状态
        mujoco.mj_step(model, data)
        # 定义视口的大小和位置
        viewport = mujoco.MjrRect(0, 0, 1200, 900)
        # 更新场景对象，将模型的最新状态反映到场景中
        mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        # 将场景渲染到视口中
        mujoco.mjr_render(viewport, scene, con)

        # 交换前后缓冲区，将渲染结果显示在窗口上
        glfw.swap_buffers(window)
        # 处理所有待处理的事件，如鼠标、键盘事件等
        glfw.poll_events()

    # 终止 GLFW 库，释放相关资源
    glfw.terminate()

if __name__ == "__main__":
    # 当脚本作为主程序运行时，调用 main 函数
    main()