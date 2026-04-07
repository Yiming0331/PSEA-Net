import argparse
import copy
import os
import random

import cv2
from tqdm import tqdm

# from distortions import (block_wise, color_contrast, color_saturation,
#                          gaussian_blur, gaussian_noise_color, jpeg_compression)
from distortions import (block_wise, color_contrast, color_saturation,
                         gaussian_blur, gaussian_noise_color, jpeg_compression)

def parse_args():
    parser = argparse.ArgumentParser(description='Add a distortion to video.')
    # --vid_in: 输入视频路径（必须提供）
    parser.add_argument('--vid_in',
                        type=str,
                        required=True,
                        help='path to the input video')
    # --vid_out: 输出视频路径（必须提供）
    parser.add_argument('--vid_out',
                        type=str,
                        required=True,
                        help='path to the output video')
    """
    --type: 失真类型，包括：
        CS: 可能是 Color Saturation(色彩饱和度)
        CC: 可能是 Color Contrast(色彩对比度)
        BW: Black & White(黑白)
        GNC: Gaussian Noise(高斯噪声)
        GB: Gaussian Blur(高斯模糊)
        JPEG: JPEG 压缩伪影
        VC: Video Compression(视频压缩)
        random: 随机选择
        --level: 失真强度级别(1-5,数字越大失真越严重)
    """
    parser.add_argument(
        '--type',
        type=str,
        default='random',
        help='distortion type: CS | CC | BW | GNC | GB | JPEG | VC | random')
    parser.add_argument('--level',
                        type=str,
                        default='random',
                        help='distortion level: 1 | 2 | 3 | 4 | 5 | random')
    # 元数据文件输出路径，用于保存处理信息
    parser.add_argument('--meta_path',
                        type=str,
                        default=None,
                        help='path to the output video meta file')
    # --via_xvid: 布尔标志，如果指定则先输出为 XVID AVI 格式，再用 ffmpeg 转换
    parser.add_argument(
        '--via_xvid',
        action='store_true',
        help='if add this argument, write to XVID .avi video first, '
        "then convert it to 'vid_out' by ffmpeg.")
    args = parser.parse_args()

    return args


def get_distortion_parameter(type, level):
    param_dict = dict()  # a dict of list  # 创建一个空字典，用于存储各种失真类型的参数列表
    param_dict['CS'] = [0.4, 0.3, 0.2, 0.1, 0.0]  # smaller, worse 色彩饱和度参数，值越小失真越严重
    param_dict['CC'] = [0.85, 0.725, 0.6, 0.475, 0.35]  # smaller, worse 色彩对比度参数，值越小失真越严重
    param_dict['BW'] = [16, 32, 48, 64, 80]  # larger, worse 分块失真参数，值越大失真越严重
    param_dict['GNC'] = [0.001, 0.002, 0.005, 0.01, 0.05]  # larger, worse 高斯噪声参数，值越大失真越严
    param_dict['GB'] = [7, 9, 13, 17, 21]  # larger, worse 高斯模糊参数（可能是核大小），值越大失真越严重
    param_dict['JPEG'] = [2, 3, 4, 5, 6]  # larger, worse JPEG压缩质量参数，值越大质量越差
    #param_dict['VC'] = [30, 32, 35, 38, 40]  # larger, worse 视频压缩参数（可能是CRF值），值越大质量越差

    # 由于级别从1开始（用户输入），但列表索引从0开始，所以需要减1
    # level starts from 1, list starts from 0
    return param_dict[type][level - 1]


def get_distortion_function(type):
    func_dict = dict()  # a dict of function # 创建一个空字典，用于存储失真类型对应的处理函数
    func_dict['CS'] = color_saturation # 色彩饱和度处理函数
    func_dict['CC'] = color_contrast # 色彩对比度处理函数
    func_dict['BW'] = block_wise # 分块失真处理函数
    func_dict['GNC'] = gaussian_noise_color # 彩色高斯噪声处理函数
    func_dict['GB'] = gaussian_blur # 高斯模糊处理函数
    func_dict['JPEG'] = jpeg_compression # JPEG压缩处理函数
    #func_dict['VC'] = video_compression # 视频压缩处理函数

    return func_dict[type] # 返回对应类型的处理函数


def apply_distortion_log(type, level):
    # 应用失真日志输出
    # 根据失真类型输出相应的处理日志信息
    if type == 'CS':
        print(f'Apply level-{level} color saturation change distortion...')
    elif type == 'CC':
        print(f'Apply level-{level} color contrast change distortion...')
    elif type == 'BW':
        print(f'Apply level-{level} local block-wise distortion...')
    elif type == 'GNC':
        print(f'Apply level-{level} white Gaussian noise in color components '
              'distortion...')  # 多行字符串，处理彩色分量中的高斯噪声
    elif type == 'GB':
        print(f'Apply level-{level} Gaussian blur distortion...')
    elif type == 'JPEG':
        print(f'Apply level-{level} JPEG compression distortion...')
    elif type == 'VC':
        print(f'Apply level-{level} video compression distortion...')

# 这是一个视频失真处理的主要函数，
def distortion_vid(vid_in,
                   vid_out,
                   type='random',
                   level='random',
                   via_xvid=False):
    """
    vid_in: 输入视频路径
    vid_out: 输出视频路径
    type: 失真类型，默认随机选择
    level: 失真级别，默认随机选择
    via_xvid: 是否使用 XVID 中间格式
    """
    # create output root
    # 获取输出路径的目录部分
    root = os.path.split(vid_out)[0]
    # 如果目录为空，使用当前目录
    root = '.' if root == '' else root
    # 创建目录（如果不存在）
    os.makedirs(root, exist_ok=True)

    # get distortion type
    if type == 'random':
        # 所有可选的失真类型
        dist_types = ['CS', 'CC', 'BW', 'GNC', 'GB', 'JPEG', 'VC']
        # 随机选择索引（0-6）
        type_id = random.randint(0, 6)
        # 获取随机选择的失真类型
        dist_type = dist_types[type_id]
    else:
        # 使用指定的类型
        dist_type = type

    # get distortion level
    if level == 'random':
        # 随机选择级别（1-5）
        dist_level = random.randint(1, 5)
    else:
        # 使用指定的级别
        dist_level = int(level)

    # get distortion parameter
    # # 获取具体参数值
    dist_param = get_distortion_parameter(dist_type, dist_level)

    # get distortion function
    # 获取处理函数
    dist_function = get_distortion_function(dist_type)

    # apply distortion
    # 视频压缩（VC）特殊处理
    if dist_type == 'VC':
        # # 输出日志
        apply_distortion_log(dist_type, dist_level)
        # # 直接处理整个视频
        dist_function(vid_in, vid_out, dist_param)
    else:
        # extract frames
        # # 提取视频信息
        # # 打开输入视频
        vid = cv2.VideoCapture(vid_in)
        # # 获取帧率
        fps = vid.get(cv2.CAP_PROP_FPS)
        # 获取编码格式
        fourcc = int(vid.get(cv2.CAP_PROP_FOURCC))
        # 获取宽度
        w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 获取高度
        h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 获取总帧数
        frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        # 打印视频信息
        print(f'Input video fps: {fps}')
        print(f'Input video fourcc: {fourcc}')
        print(f'Input video frame size: {w} * {h}')
        print(f'Input video frame count: {frame_count}')
        print('Extracting frames...')
        # 提取所有帧
        frame_list = []
        while True:
            success, frame = vid.read() # 读取一帧
            if not success:
                break
            frame_list.append(frame) # 添加到帧列表
        vid.release()
        assert len(frame_list) == frame_count # 验证帧数是否正确

        # add distortion to the frame and write to the new video at 'vid_out'
        # 创建视频写入器
        if via_xvid:
            # 使用 XVID 编码的临时文件
            writer = cv2.VideoWriter(
                f'{vid_out[:-4]}_tmp.avi',
                cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, (w, h))
        else:
            # 使用原始编码格式
            writer = cv2.VideoWriter(vid_out, fourcc, fps, (w, h))
        # 应用失真处理
        apply_distortion_log(dist_type, dist_level) # # 输出处理日志
        for frame in tqdm(frame_list): # 使用进度条遍历所有帧
            new_frame = dist_function(frame, dist_param) # 对每帧应用失真
            writer.write(new_frame) # 写入处理后的帧
        writer.release()  # 释放写入器
        if via_xvid:
            cmd = f'ffmpeg -i {vid_out[:-4]}_tmp.avi -y {vid_out}'  # 使用ffmpeg转换格式
            os.system(cmd)  # 执行转换命令
    # 清理临时文件
    if os.path.exists(f'{vid_out[:-4]}_tmp.avi'):
        os.remove(f'{vid_out[:-4]}_tmp.avi')
    print('Finished.')
    #  # 返回实际使用的失真类型和级别
    return dist_type, dist_level


def write_to_meta_file(meta_path, vid_in, vid_out, dist_type, dist_level):
    # create meta root
    """
    meta_path: 元数据文件路径
    vid_in: 输入视频路径
    vid_out: 输出视频路径
    dist_type: 失真类型
    dist_level: 失真级别
    """
    root = os.path.split(meta_path)[0] # 获取元数据文件所在的目录路径
   
    root = '.' if root == '' else root # 如果目录为空字符串，使用当前目录
    os.makedirs(root, exist_ok=True) # 创建目录（如果不存在）

    # 创建一个空字典，用于存储所有视频的元数据
    meta_dict = dict()  # a dict of list
    # if exist, get original meta
    # # 如果元数据文件已存在，读取现有内容
    ## 以只读方式打开文件
    if os.path.exists(meta_path):
        f = open(meta_path, 'r')
        # # 读取所有行并移除换行符
        lines = f.read().splitlines()
        # # 关闭文件
        f.close()
        
        # 解析每一行数据
        for l in lines:
            # 分割每行：第一个元素是视频路径，剩余的是失真元数据
            vid_path, dist_meta = l.split()[0], l.split()[1:]
            # # 将视频路径作为键，失真列表作为值存入字典
            meta_dict[vid_path] = dist_meta

    # update meta
    # 获取输入视频的现有失真记录（如果存在），否则创建空列表
    meta_list = copy.deepcopy(meta_dict[vid_in]) if vid_in in meta_dict else []
    # 添加新的失真记录，格式为 "类型:级别"
    meta_list.append(f'{dist_type}:{dist_level}')
    # 将输出视频路径和更新后的失真列表存入字典
    meta_dict[vid_out] = meta_list

    # write meta
    # 以写入模式打开文件（会覆盖原有内容）
    f = open(meta_path, 'w')
    # # 遍历字典中的所有条目
    for k, v in meta_dict.items():
        # 将视频路径和失真列表合并为一个字符串，用空格分隔
        f.write(' '.join([k] + v) + '\n') # 写入一行并添加换行符
    f.close() # 关闭文件


def main():
    args = parse_args()

    vid_in = args.vid_in
    vid_out = args.vid_out
    type = args.type
    level = args.level
    meta_path = args.meta_path
    via_xvid = args.via_xvid

    # check input args
    assert os.path.exists(vid_in), 'Input video does not exist.'
    assert vid_in != vid_out, ('Paths to the input and output videos '
                               'should NOT be the same.')
    type_list = ['CS', 'CC', 'BW', 'GNC', 'GB', 'JPEG', 'VC', 'random']
    if type not in type_list:
        raise ValueError(
            f"Expect distortion type in {type_list}, but got '{type}'.")
    level_list = ['1', '2', '3', '4', '5', 'random']
    if level not in level_list:
        raise ValueError(
            f"Expect distortion level in {level_list}, but got '{level}'.")

    # add distortion to the input video and write to 'vid_out'
    dist_type, dist_level = distortion_vid(vid_in, vid_out, type, level,
                                           via_xvid)

    # if meta_path is not None, write meta
    if meta_path is not None:
        # write to meta file
        write_to_meta_file(meta_path, vid_in, vid_out, dist_type, dist_level)


if __name__ == '__main__':
    main()
