import glob
import json
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode 
bicubic = InterpolationMode.BICUBIC
from torch import nn

import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import copy
import gzip
from torchvision.transforms.functional import affine

def get_param2img(opt, device='cuda'):
    '''
    获取参数到图像的转换函数
    
    该函数负责加载预训练的笔画参数到图像模型，并返回一个前向传播函数，用于将笔画参数转换为实际的笔画图像。
    
    参数:
        opt: 包含各种配置参数的选项对象
        device: 运行模型的设备，默认为'cuda'
        
    返回:
        forward: 一个函数，接受笔画参数和目标渲染尺寸，返回渲染后的笔画图像
    '''
    
    # 获取画布尺寸（以米为单位）
    w_canvas_m = opt.CANVAS_WIDTH_M
    h_canvas_m = opt.CANVAS_HEIGHT_M
    
    # 加载param2image模型输出表示的米数（从设置文件中读取）
    with open(os.path.join(opt.cache_dir, 'param2stroke_settings.json'), 'r') as f:
        settings = json.load(f)
        # print('Param2Stroke Settings:', settings)
        w_p2i_m = settings['w_p2i_m']  # 模型输出宽度（米）
        h_p2i_m = settings['h_p2i_m']  # 模型输出高度（米）
        xtra_room_horz_m = settings['xtra_room_horz_m']  # 水平方向额外空间（米）
        xtra_room_vert_m = settings['xtra_room_vert_m']  # 垂直方向额外空间（米）
        MAX_BEND = settings['MAX_BEND']  # 最大弯曲程度
    
    # 验证画布宽度是否足够大，确保能够正确渲染笔画
    # 如果模型输出宽度减去水平额外空间大于画布宽度的一半，则画布太小
    if (w_p2i_m- xtra_room_horz_m) > (0.5 * w_canvas_m):
        print("w_p2i_m", w_p2i_m, "xtra_room_horz_m", xtra_room_horz_m, "w_canvas_m", w_canvas_m, "0.5 * w_canvas_m", 0.5 * w_canvas_m)
        raise Exception("画布宽度小于最大笔画长度的两倍，这使得渲染变得非常困难。必须使用更大的画布。")
    
    # 初始化StrokeParametersToImage模型并加载预训练权重
    param2img = StrokeParametersToImage()
    param2img.load_state_dict(torch.load(
        os.path.join(opt.cache_dir, 'param2img.pt')))
    param2img.eval()  # 设置为评估模式
    param2img.to(device)  # 将模型移至指定设备

    def forward(param, h_render_pix, w_render_pix):
        '''
        前向传播函数，将笔画参数转换为图像
        
        参数:
            param: 笔画参数张量
            h_render_pix: 目标渲染高度（像素）
            w_render_pix: 目标渲染宽度（像素）
            
        返回:
            渲染后的笔画图像
        '''
        # 根据目标渲染尺寸计算param2image输出应该调整的大小
        # 将米转换为像素的比例关系
        w_p2i_render_pix = int((w_p2i_m / w_canvas_m) * w_render_pix)
        h_p2i_render_pix = int((h_p2i_m / h_canvas_m) * h_render_pix)
        # 创建调整大小的变换
        res_to_render = transforms.Resize((h_p2i_render_pix, w_p2i_render_pix), bicubic, antialias=True)

        # 对param2image的输出进行填充，使笔画的起始点位于画布中央，并且图像尺寸与渲染尺寸匹配
        # 计算填充量（以米为单位）
        pad_left_m = 0.5 * w_canvas_m - xtra_room_horz_m  # 左侧填充（米）
        pad_right_m = w_canvas_m - pad_left_m - w_p2i_m  # 右侧填充（米）
        pad_top_m = 0.5 * h_canvas_m - MAX_BEND - xtra_room_vert_m  # 顶部填充（米）
        pad_bottom_m = 0.5 * h_canvas_m - MAX_BEND - xtra_room_vert_m  # 底部填充（米）

        # 将填充量从米转换为像素
        pad_left_pix =   int(pad_left_m   * (w_render_pix / w_canvas_m))
        pad_right_pix =  int(pad_right_m  * (w_render_pix / w_canvas_m))
        pad_top_pix =    int(pad_top_m    * (h_render_pix / h_canvas_m))
        pad_bottom_pix = int(pad_bottom_m * (h_render_pix / h_canvas_m))

        # 创建填充变换
        pad_for_full = transforms.Pad((pad_left_pix, pad_top_pix, pad_right_pix, pad_bottom_pix))

        # 应用模型和变换：先通过模型生成笔画，然后调整大小，最后进行填充
        return pad_for_full(res_to_render(param2img(param)))
    
    # 返回前向传播函数
    return forward  # 返回forward函数而不是模型本身



def special_sigmoid(x):
    '''
    特殊的Sigmoid函数，用于增强笔画的对比度
    
    参数:
        x: 输入张量
        
    返回:
        应用特殊Sigmoid函数后的张量
    '''
    # 使用自定义参数的Sigmoid函数，增强对比度和锐度
    # 公式: 1/(1+exp(-1*((x*2-1)+0.2)/0.05))
    # 相比标准Sigmoid，这个函数有更陡峭的曲线，产生更清晰的边缘
    return 1/(1+torch.exp(-1.*((x*2-1)+0.2) / 0.05))
    # return x

    # x[x < 0.1] = 1/(1+torch.exp(-1.*((x[x < 0.1]*2-1)+0.2) / 0.05))
    # return x

def get_n_params(model):
    '''
    计算模型的参数总数
    
    参数:
        model: PyTorch模型
        
    返回:
        模型的参数总数
    '''
    pp = 0  # 参数计数器
    # 遍历模型的所有参数
    for p in list(model.parameters()):
        nn = 1  # 当前参数的元素数量
        # 计算参数的元素总数（所有维度的乘积）
        for s in list(p.size()):
            nn = nn * s
        # 累加到总参数计数
        pp += nn 
    return pp


def to_full_param(length, bend, z, alpha=0.0, device='cuda'):
    '''
    创建完整的笔画参数向量
    
    参数:
        length: 笔画长度
        bend: 笔画弯曲度
        z: 笔画粗细
        alpha: 笔画透明度，默认为0.0
        device: 运行设备，默认为'cuda'
        
    返回:
        包含所有参数的张量，形状为[1,4]
    '''
    # 创建一个形状为[1,4]的零张量
    full_param = torch.zeros((1,4)).to(device)
    
    # 设置各个参数值
    full_param[0,0] = length  # 第一个参数：笔画长度
    full_param[0,1] = bend    # 第二个参数：笔画弯曲度
    full_param[0,2] = z       # 第三个参数：笔画粗细
    full_param[0,3] = alpha   # 第四个参数：笔画透明度

    return full_param

def process_img(img):
    '''
    处理图像用于可视化
    
    参数:
        img: 输入图像张量
        
    返回:
        处理后的NumPy数组，值范围为[0,255]
    '''
    # 将张量转换为NumPy数组，限制值范围在[0,1]之间，然后缩放到[0,255]范围
    return np.clip(img.detach().cpu().numpy(), a_min=0, a_max=1)*255

def log_all_permutations(model, writer, opt):
    '''
    记录并可视化所有参数组合的笔画效果
    
    该函数生成三种不同参数组合的可视化图像：
    1. 长度vs弯曲度（固定粗细和透明度）
    2. 长度vs粗细（固定弯曲度和透明度）
    3. 长度vs透明度（固定弯曲度和粗细）
    
    参数:
        model: 训练好的StrokeParametersToImage模型
        writer: TensorBoard写入器，用于记录可视化结果
        opt: 包含各种配置参数的选项对象
    '''
    # 设置图像网格大小
    n_img = 5
    
    # 为每个参数创建均匀分布的值范围
    lengths = torch.linspace(opt.MIN_STROKE_LENGTH, opt.MAX_STROKE_LENGTH, steps=n_img)  # 笔画长度范围
    bends = torch.linspace(-1.0*opt.MAX_BEND, opt.MAX_BEND, steps=n_img)  # 笔画弯曲度范围
    zs = torch.linspace(opt.MIN_STROKE_Z, 1.0, steps=n_img)  # 笔画粗细范围
    alphas = torch.linspace(-1.*opt.MAX_ALPHA, opt.MAX_ALPHA, steps=n_img)  # 笔画透明度范围

    # 1. 可视化长度vs弯曲度（固定粗细为0.5，透明度为默认值）
    whole_thing = []
    for i in range(n_img):  # 遍历不同长度
        row = []
        for j in range(n_img):  # 遍历不同弯曲度
            # 创建参数向量
            trajectory = to_full_param(lengths[i], bends[j], 0.5)
            # 生成笔画图像并应用特殊sigmoid
            s = 1-special_sigmoid(model(trajectory))
            # 将张量转换为NumPy数组并限制值范围
            s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)
            row.append(s)
        # 水平连接一行中的所有图像
        whole_thing.append(np.concatenate(row, axis=1))
    # 垂直连接所有行，形成完整的网格图像
    whole_thing = np.concatenate(whole_thing, axis=0)
    
    # 创建并保存图像
    fig, ax = plt.subplots(1, 1, figsize=(10,12))    
    ax.imshow(whole_thing, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    # 将图像添加到TensorBoard
    writer.add_figure('images_stroke_modeling/bend_vs_length', fig, 0)

    # 2. 可视化长度vs粗细（固定弯曲度为0.0，透明度为默认值）
    whole_thing = []
    for i in range(n_img):  # 遍历不同长度
        row = []
        for j in range(n_img):  # 遍历不同粗细
            trajectory = to_full_param(lengths[i], 0.0, zs[j])
            s = 1-special_sigmoid(model(trajectory))
            s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)
            row.append(s)
        whole_thing.append(np.concatenate(row, axis=1))
    whole_thing = np.concatenate(whole_thing, axis=0)
    
    # 创建并保存图像
    fig, ax = plt.subplots(1, 1, figsize=(10,12))    
    ax.imshow(whole_thing, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    writer.add_figure('images_stroke_modeling/thickness_vs_length', fig, 0)

    # 3. 可视化长度vs透明度（固定弯曲度为0.0，粗细为0.5）
    whole_thing = []
    for i in range(n_img):  # 遍历不同长度
        row = []
        for j in range(n_img):  # 遍历不同透明度
            trajectory = to_full_param(lengths[i], 0.0, 0.5, alphas[j])
            s = 1-model(trajectory)  # 注意这里没有使用special_sigmoid
            s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)
            row.append(s)
        whole_thing.append(np.concatenate(row, axis=1))
    whole_thing = np.concatenate(whole_thing, axis=0)
    
    # 创建并保存图像
    fig, ax = plt.subplots(1, 1, figsize=(10,12))    
    ax.imshow(whole_thing, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    writer.add_figure('images_stroke_modeling/length_vs_alpha', fig, 0)

def remove_background_noise(strokes):
    '''
    移除笔画图像中的背景噪声
    
    该函数通过识别图像中不太可能是笔画的区域，并将这些区域的值设为0，
    从而清理数据集中的感知问题和非笔画数据。
    
    参数:
        strokes: 笔画图像张量集合
        
    返回:
        清理后的笔画图像张量集合
    '''
    # 清除不太可能是笔画的区域
    # 即移除非笔画数据，解决感知问题
    # print('mean', strokes.mean())
    from scipy import ndimage
    
    # 计算所有笔画的平均值，得到一个热力图
    stroke_mean = strokes.mean(dim=0)
    # plt.imshow(stroke_mean.cpu().numpy())
    # plt.colorbar()
    # plt.show()

    # 应用最大值滤波，增强笔画区域
    stroke_mean = ndimage.maximum_filter(stroke_mean, size=30)
    stroke_mean = torch.from_numpy(stroke_mean)
    
    # plt.imshow(stroke_mean)
    # plt.colorbar()
    # plt.show()
    # print(torch.quantile(stroke_mean, 0.1))
    
    # 识别不太可能是笔画的区域（值低于中位数的区域）
    unlikely_areas = (stroke_mean < torch.quantile(stroke_mean[stroke_mean > 0.001], 0.5))#[None,:,:]
    # plt.imshow(unlikely_areas*0.5 + strokes.mean(dim=0))
    # plt.colorbar()
    # plt.show()
    
    # 应用最小值滤波，略微扩大识别的区域
    unlikely_areas = ndimage.minimum_filter(unlikely_areas, size=50) # 略微扩大区域
    unlikely_areas = torch.from_numpy(unlikely_areas)

    # plt.imshow(unlikely_areas)
    # plt.colorbar()
    # plt.show()

    # plt.imshow(unlikely_areas*0.5 + strokes.mean(dim=0))
    # plt.colorbar()
    # plt.show()
    
    # 将识别出的非笔画区域设为0
    strokes[:,unlikely_areas] = 0
    # print('mean', strokes.mean())
    return strokes

def log_images(imgs, labels, label, writer, step=0):
    '''
    记录图像到TensorBoard
    
    参数:
        imgs: 图像列表
        labels: 每个图像的标签列表
        label: 整个图像组的标签
        writer: TensorBoard写入器
        step: 步骤编号，默认为0
    '''
    # 创建包含多个子图的图像
    fig, ax = plt.subplots(1, len(imgs), figsize=(5*len(imgs),5))

    # 遍历所有图像并添加到子图中
    for i in range(len(imgs)):
        # print(imgs[i].min(), imgs[i].max())
        # 显示图像，使用灰度颜色映射，值范围为[0,255]
        ax[i].imshow(imgs[i], cmap='gray', vmin=0, vmax=255)
        # 移除坐标轴刻度
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        # 设置子图标题
        ax[i].set_title(labels[i])
    # 调整布局
    fig.tight_layout()
    # 将图像添加到TensorBoard
    writer.add_figure(label, fig, step)

class StrokeParametersToImage(nn.Module):
    '''
    笔画参数到图像转换模型
    
    该模型将笔画的参数（长度、弯曲度、粗细和透明度）转换为对应的笔画图像。
    模型由两部分组成：
    1. 全连接网络：将参数转换为特征向量
    2. 卷积网络：将特征向量转换为图像
    '''
    def __init__(self):
        '''
        初始化模型结构
        '''
        super(StrokeParametersToImage, self).__init__()
        nh = 20  # 隐藏层神经元数量
        self.nc = 20  # 卷积层通道数
        self.size_x = 128  # 输出图像宽度
        self.size_y = 64   # 输出图像高度
        
        # 全连接网络部分：将4维参数向量转换为特征向量
        self.main = nn.Sequential(
            nn.BatchNorm1d(4),  # 对输入参数进行批归一化
            nn.Linear(4, nh),  # 全连接层：4维参数 -> nh维特征
            nn.LeakyReLU(0.2, inplace=True),  # 激活函数
            nn.BatchNorm1d(nh),  # 批归一化
            nn.Linear(nh, self.size_x*self.size_y),  # 全连接层：nh维特征 -> 图像尺寸
            nn.LeakyReLU(0.2, inplace=True)  # 激活函数
        )
        
        # 卷积网络部分：将特征向量转换为图像
        self.conv = nn.Sequential(
            nn.BatchNorm2d(1),  # 对输入特征图进行批归一化
            nn.Conv2d(1, self.nc, kernel_size=5, padding='same', dilation=1),  # 卷积层
            nn.LeakyReLU(0.2, inplace=True),  # 激活函数
            nn.BatchNorm2d(self.nc),  # 批归一化
            nn.Conv2d(self.nc, 1, kernel_size=5, padding='same', dilation=1),  # 卷积层
            nn.Sigmoid()  # Sigmoid激活函数，输出值范围为[0,1]
        )

    def forward(self, x):
        '''
        前向传播函数
        
        参数:
            x: 笔画参数张量，形状为[batch_size, 4]
               4个参数分别为：长度、弯曲度、粗细和透明度
               
        返回:
            笔画图像张量，形状为[batch_size, size_y, size_x]
        '''
        # 1. 通过全连接网络将参数转换为特征向量
        # 2. 将特征向量重塑为2D图像形状
        # 3. 通过卷积网络处理图像
        # 4. 提取第一个通道作为输出图像
        x = self.conv((self.main(x).view(-1, 1, self.size_y, self.size_x)))[:,0]
        return x
    


l1_loss = nn.L1Loss()  # L1损失函数，用于计算绝对误差

def shift_invariant_loss(pred, real, n=8, delta=0.02):
    '''
    位移不变损失函数
    
    该函数通过尝试多种小的位移变换，找到预测图像与真实图像之间的最佳匹配位置，
    从而使模型对小的位置偏移保持鲁棒性。这对于笔画渲染特别重要，因为笔画的精确位置
    可能会有细微变化，但整体形状和外观应保持一致。
    
    参数:
        pred: 预测的笔画图像
        real: 真实的笔画图像
        n: 每个方向上尝试的位移数量，默认为8
        delta: 最大位移比例，默认为0.02（相对于图像尺寸）
        
    返回:
        组合损失值：最小位移损失 + L1损失
    '''
    losses = None
    # 在水平和垂直方向上尝试不同的小位移
    for dx in torch.linspace(start=-1.0*delta, end=delta, steps=n):  # 水平位移范围
        for dy in torch.linspace(start=-1.0*delta, end=delta, steps=n):  # 垂直位移范围
            # 计算像素级位移
            x = int(dx*real.shape[2])  # 水平位移像素数
            y = int(dy*real.shape[1])  # 垂直位移像素数
            
            # 对预测图像应用仿射变换，进行平移
            translated_pred = affine(pred, angle=0, translate=(x, y), fill=0, scale=1.0, shear=0)

            # 计算L2损失（均方误差）
            diff = (translated_pred - real)**2
            l = diff.mean(dim=(1,2))  # 对每个样本计算平均损失
            
            # 收集所有位移组合的损失
            losses = l[None,:] if losses is None else torch.cat([losses, l[None,:]], dim=0)

    # 只使用产生最小损失值的位移组合
    # 这样可以找到预测图像与真实图像之间的最佳匹配位置
    loss, inds = torch.min(losses, dim=0)
    
    # 返回最小位移损失的平均值 + L1损失
    # L1损失确保整体亮度和对比度的一致性
    return loss.mean() + l1_loss(pred, real)

def train_param2stroke(opt, device='cuda'):
    # param2img = get_param2img(opt)
    # x = param2img(torch.zeros(1,4, device=device), 200, 400)
    # print(x.shape)
    # plt.imshow(x[0].cpu().detach().numpy())
    # plt.show()
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Load data
    stroke_img_fns = glob.glob(os.path.join(opt.cache_dir, 'stroke_library', 'stroke_intensities*.npy'))
    stroke_param_fns = glob.glob(os.path.join(opt.cache_dir, 'stroke_library', 'stroke_parameters*.npy'))
    stroke_img_fns = sorted(stroke_img_fns)
    stroke_param_fns = sorted(stroke_param_fns)

    strokes = None
    for stroke_img_fn in stroke_img_fns:
        with gzip.GzipFile(stroke_img_fn,'r') as f:
            s = np.load(f, allow_pickle=True).astype(np.float32)/255.

            h_og, w_og = s[0].shape[0], s[0].shape[1]
            # Crop so that we only predict around the area around the stroke not all the background blank pix
            xtra_room_horz_m = 0.01 # Added padding to ensure we don't crop out actual paint
            xtra_room_vert_m = 0.001
            # Do in meters first. ws_m = width start crop in meters
            ws_m = (opt.STROKE_LIBRARY_CANVAS_WIDTH_M/2) - xtra_room_horz_m
            we_m = ws_m + xtra_room_horz_m*2 + opt.MAX_STROKE_LENGTH
            hs_m = (opt.STROKE_LIBRARY_CANVAS_HEIGHT_M/2) - opt.MAX_BEND - xtra_room_vert_m
            he_m = hs_m + 2*xtra_room_vert_m + 2*opt.MAX_BEND
            # Convert from meters to pix
            pix_per_m = w_og / opt.STROKE_LIBRARY_CANVAS_WIDTH_M
            ws, we, hs, he = ws_m*pix_per_m, we_m*pix_per_m, hs_m*pix_per_m, he_m*pix_per_m
            ws, we, hs, he = int(ws), int(we), int(hs), int(he)
            # print(ws, we, hs, he)
            s = s[:, hs:he, ws:we]

            strokes = s if strokes is None else np.concatenate([strokes, s])

    parameters = None
    for stroke_param_fn in stroke_param_fns:
        p = np.load(stroke_param_fn, allow_pickle=True, encoding='bytes') 
        parameters = p if parameters is None else np.concatenate([parameters, p]) 
    
    # with gzip.GzipFile(os.path.join(opt.cache_dir, 'stroke_intensities.npy'),'r') as f:
    #     strokes = np.load(f).astype(np.float32)/255.
    # parameters = np.load(os.path.join(opt.cache_dir, 'stroke_parameters.npy'), 
    #         allow_pickle=True, encoding='bytes') 

    strokes = torch.from_numpy(strokes).float().nan_to_num()
    parameters = torch.from_numpy(parameters.astype(np.float32)).float().nan_to_num()
    
    n = len(strokes)

    # Randomize
    rand_ind = torch.randperm(strokes.shape[0])
    strokes = strokes[rand_ind]
    parameters = parameters[rand_ind]

    # Discrete. Makes the model push towards making bolder strokes
    strokes[strokes >= 0.5] = 1.
    strokes[strokes < 0.5] = 0. # Make sure background is very much gone


    

    # Save the amount of meters that the output of the param2image model represents
    w_p2i_m = we_m - ws_m 
    h_p2i_m = he_m - hs_m 
    with open(os.path.join(opt.cache_dir, 'param2stroke_settings.json'), 'w') as f:
        settings = {}
        settings['w_p2i_m'] = w_p2i_m
        settings['h_p2i_m'] = h_p2i_m
        settings['xtra_room_horz_m'] = xtra_room_horz_m
        settings['xtra_room_vert_m'] = xtra_room_vert_m
        settings['MAX_BEND'] = opt.MAX_BEND
        json.dump(settings, f, indent=4)


    strokes = remove_background_noise(strokes)

    # Resize strokes the size they'll be predicted at
    t = StrokeParametersToImage()
    strokes = transforms.Resize((t.size_y,t.size_x), bicubic, antialias=True)(strokes)

    # for i in range(len(strokes)):
    #     strokes[i] -= strokes[i].min()
    #     if strokes[i].max() > 0.01:
    #         strokes[i] /= strokes[i].max()
    #     # strokes[i] *= 0.95
    #     # print(strokes[i].min(), strokes[i].max())
    
    # Filter out strokes that are bad perception. Avg is too high.
    # One bad apple can really spoil the bunch
    good_strokes = []
    good_parameters = []
    for i in range(len(strokes)):
        if strokes[i].mean() < 0.4: 
            good_strokes.append(strokes[i])
            good_parameters.append(parameters[i])
    print(len(strokes)- len(good_strokes), 'strokes removed because average value too high')
    strokes = torch.stack(good_strokes, dim=0)
    parameters = torch.stack(good_parameters, dim=0)

    h, w = strokes[0].shape[0], strokes[0].shape[1]

    strokes = strokes.to(device)
    parameters = parameters.to(device)

    trans = StrokeParametersToImage() 
    trans = trans.to(device)
    print('# parameters in Param2Image model:', get_n_params(trans))
    optim = torch.optim.Adam(trans.parameters(), lr=1e-3)#, weight_decay=1e-5)
    best_model = copy.deepcopy(trans)
    best_val_loss = 99999
    best_hasnt_changed_for = 0

    val_prop = .2

    train_strokes = strokes[int(val_prop*n):]
    train_parameters = parameters[int(val_prop*n):]

    val_strokes = strokes[:int(val_prop*n)]
    val_parameters = parameters[:int(val_prop*n)]
    print('{} training strokes. {} validation strokes'.format(len(train_strokes), len(val_strokes)))

    param_stds = train_parameters.std(dim=0)

    for it in tqdm(range(5000)):
        if best_hasnt_changed_for >= 200 and it > 200:
            break # all done :)
        optim.zero_grad()

        noise = torch.randn(train_parameters.shape).to(device)*param_stds[None,:]*0.15 # For robustness
        pred_strokes = trans(train_parameters + noise)

        loss = shift_invariant_loss(pred_strokes, train_strokes)
        # loss = nn.MSELoss()(pred_strokes, train_strokes) # MSE loss produces crisper stroke images

        ep_loss = loss.item()
        loss.backward()
        optim.step()

        opt.writer.add_scalar('loss/train_loss_stroke_model', ep_loss, it)
        
        n_view = 10
        if it % 5 == 0:
            with torch.no_grad():
                trans.eval()
                pred_strokes_val = trans(val_parameters)

                loss = shift_invariant_loss(pred_strokes_val, val_strokes)
                if it % 15 == 0: 
                    opt.writer.add_scalar('loss/val_loss_stroke_model', loss.item(), it)
                if loss.item() < best_val_loss and it > 50:
                    best_val_loss = loss.item()
                    best_hasnt_changed_for = 0
                    best_model = copy.deepcopy(trans)
                best_hasnt_changed_for += 5
                trans.train()


    with torch.no_grad():
        def draw_grid(image, line_space_x=20, line_space_y=20):
            H, W = image.shape
            # image[0:H:line_space_x] = 0
            # image[:, 0:W:line_space_y] = 0
            return image
        best_model.eval()
        pred_strokes_val = best_model(val_parameters)
        real_imgs, pred_imgs = None, None
        for val_ind in range(min(n_view,len(val_strokes))):
            l, b, z, alpha = val_parameters[val_ind][0], val_parameters[val_ind][1], val_parameters[val_ind][2], val_parameters[val_ind][3]
            # log_images([process_img(1-val_strokes[val_ind]),
            #     process_img(1-special_sigmoid(pred_strokes_val[val_ind]))], 
            #     ['real','sim'], 'images_stroke_modeling_stroke/val_{}_sim_stroke_best_b{:.2f}_l{:.2f}_z{:.2f}_alph{:.2f}'.format(
            #                 val_ind, b, l, z, alpha), opt.writer)
            pred_img = draw_grid(1-special_sigmoid(pred_strokes_val[val_ind]))
            real_img = draw_grid(1-val_strokes[val_ind])
            real_imgs = real_img if real_imgs is None else torch.cat([real_imgs, real_img], dim=0)
            pred_imgs = pred_img if pred_imgs is None else torch.cat([pred_imgs, pred_img], dim=0)
        real_imgs[:,:5] = 0
        pred_imgs[:,:5] = 0
        whole_img = torch.cat([real_imgs, pred_imgs], dim=1)
        # whole_img = draw_grid(whole_img)
        opt.writer.add_image('images_stroke_modeling_stroke/val', process_img(whole_img), 0)


        pred_strokes_train = best_model(train_parameters)
        real_imgs, pred_imgs = None, None
        for train_ind in range(min(n_view,len(train_strokes))):
            l, b, z, alpha = train_parameters[train_ind][0], train_parameters[train_ind][1], train_parameters[train_ind][2], train_parameters[train_ind][3]
            # log_images([process_img(1-train_strokes[train_ind]),
            #     process_img(1-special_sigmoid(pred_strokes_train[train_ind]))], 
            #     ['real','sim'], 'images_stroke_modeling_stroke/train_{}_sim_stroke_best_b{:.2f}_l{:.2f}_z{:.2f}_alph{:.2f}'.format(
            #                 train_ind, b, l, z, alpha), opt.writer)
            pred_img = draw_grid(1-special_sigmoid(pred_strokes_train[train_ind]))
            real_img = draw_grid(1-train_strokes[train_ind])
            real_imgs = real_img if real_imgs is None else torch.cat([real_imgs, real_img], dim=0)
            pred_imgs = pred_img if pred_imgs is None else torch.cat([pred_imgs, pred_img], dim=0)
        real_imgs[:,:5] = 0
        pred_imgs[:,:5] = 0
        whole_img = torch.cat([real_imgs, pred_imgs], dim=1)
        # whole_img = draw_grid(whole_img)
        opt.writer.add_image('images_stroke_modeling_stroke/train', process_img(whole_img), 0)
        
        log_all_permutations(best_model, opt.writer, opt)
    torch.save(best_model.cpu().state_dict(), os.path.join(opt.cache_dir, 'param2img.pt'))

    return h_og, w_og


if __name__ == '__main__':
    from options import Options
    opt = Options()
    opt.gather_options()

    torch.manual_seed(0)
    np.random.seed(0)

    from paint_utils3 import create_tensorboard
    opt.writer = create_tensorboard()

    # Train brush strokes
    train_param2stroke(opt)