from loguru import logger

def parse_vox_chunks(data, offset=0):
    """
    递归解析VOX文件中的所有XYZI和RGBA块，返回颜色索引集合
    """
    used_color_indices = set()  # 用于存储所有子模型中使用的颜色索引
    palette = [(0, 0, 0, 255)] * 256  # 默认调色板，256个颜色

    while offset < len(data):
        # 读取块ID和大小
        chunk_id = data[offset:offset + 4].decode("ascii", errors="ignore")
        chunk_size = int.from_bytes(data[offset + 4:offset + 8], "little")
        children_size = int.from_bytes(data[offset + 8:offset + 12], "little")
        chunk_data_offset = offset + 12

        if chunk_id == "RGBA":
            # 读取 RGBA 块，解析调色板
            logger.debug("找到 RGBA 块")
            palette = [data[chunk_data_offset + i:chunk_data_offset + i + 4]
                       for i in range(0, 1024, 4)]

        elif chunk_id == "XYZI":
            # 读取 XYZI 块，收集颜色索引
            logger.debug("找到 XYZI 块")
            num_voxels = int.from_bytes(data[chunk_data_offset:chunk_data_offset + 4], "little")
            voxel_data_offset = chunk_data_offset + 4

            for i in range(num_voxels):
                _, _, _, color_index = data[voxel_data_offset + i * 4: voxel_data_offset + (i + 1) * 4]
                if color_index > 0:  # 忽略无效颜色索引0
                    used_color_indices.add(color_index)

        elif chunk_id == "PACK":
            # 递归处理多个子模型
            num_models = int.from_bytes(data[chunk_data_offset:chunk_data_offset + 4], "little")
            logger.info(f"找到 PACK 块，包含 {num_models} 个子模型")

        # 递归处理子块（如果存在）
        if children_size > 0:
            logger.debug(f"递归解析 {chunk_id} 的子块")
            child_offset = chunk_data_offset + chunk_size
            sub_indices, sub_palette = parse_vox_chunks(data, child_offset)
            used_color_indices.update(sub_indices)
            palette = sub_palette  # 子模型的调色板覆盖当前调色板

        # 移动到下一个块
        offset = chunk_data_offset + chunk_size + children_size

    return used_color_indices, palette


def count_recursive_vox_colors(vox_file):
    """
    统计VOX文件及其子模型中实际使用到的颜色种类数量
    """
    with open(vox_file, "rb") as f:
        data = f.read()

    # 解析VOX文件所有块
    used_color_indices, palette = parse_vox_chunks(data, 0)

    # 映射颜色索引到RGBA颜色
    used_colors = {palette[index - 1] for index in used_color_indices if 0 < index <= 256}

    logger.info(f"实际使用到的颜色种类数量: {len(used_colors)}")
    return len(used_colors)


# 使用示例
vox_file_path = r"D:\tmp\futureSciencePrize.vox"
color_count = count_recursive_vox_colors(vox_file_path)
print(f"实际使用到的颜色种类数量: {color_count}")
