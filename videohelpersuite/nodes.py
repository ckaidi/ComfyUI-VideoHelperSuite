import os
import sys
import json
import subprocess
import numpy as np
import re
import datetime
from typing import List
import torch
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo
from pathlib import Path
from string import Template
import itertools
import functools

import folder_paths
from .logger import logger
from .image_latent_nodes import *
from .load_video_nodes import LoadVideoUpload, LoadVideoPath, LoadVideoFFmpegUpload, LoadVideoFFmpegPath, LoadImagePath
from .load_images_nodes import LoadImagesFromDirectoryUpload, LoadImagesFromDirectoryPath
from .batched_nodes import VAEEncodeBatched, VAEDecodeBatched
from .utils import ffmpeg_path, get_audio, hash_path, validate_path, requeue_workflow, \
        gifski_path, calculate_file_hash, strip_path, try_download_video, is_url, \
        imageOrLatent, BIGMAX, merge_filter_args, ENCODE_ARGS, floatOrInt, cached, \
        ContainsAll
from comfy.utils import ProgressBar

if 'VHS_video_formats' not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["VHS_video_formats"] = ((),{".json"})
if len(folder_paths.folder_names_and_paths['VHS_video_formats'][1]) == 0:
    folder_paths.folder_names_and_paths["VHS_video_formats"][1].add(".json")
audio_extensions = ['mp3', 'mp4', 'wav', 'ogg']

def flatten_list(l):
    ret = []
    for e in l:
        if isinstance(e, list):
            ret.extend(e)
        else:
            ret.append(e)
    return ret

def iterate_format(video_format, for_widgets=True):
    """Provides an iterator over widgets, or arguments"""
    def indirector(cont, index):
        if isinstance(cont[index], list) and (not for_widgets
          or len(cont[index])> 1 and not isinstance(cont[index][1], dict)):
            inp = yield cont[index]
            if inp is not None:
                cont[index] = inp
                yield
    for k in video_format:
        if k == "extra_widgets":
            if for_widgets:
                yield from video_format["extra_widgets"]
        elif k.endswith("_pass"):
            for i in range(len(video_format[k])):
                yield from indirector(video_format[k], i)
            if not for_widgets:
                video_format[k] = flatten_list(video_format[k])
        else:
            yield from indirector(video_format, k)

base_formats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video_formats")
@cached(5)
def get_video_formats():
    format_files = {}
    for format_name in folder_paths.get_filename_list("VHS_video_formats"):
        format_files[format_name] = folder_paths.get_full_path("VHS_video_formats", format_name)
    for item in os.scandir(base_formats_dir):
        if not item.is_file() or not item.name.endswith('.json'):
            continue
        format_files[item.name[:-5]] = item.path
    formats = []
    format_widgets = {}
    for format_name, path in format_files.items():
        with open(path, 'r') as stream:
            video_format = json.load(stream)
        if "gifski_pass" in video_format and gifski_path is None:
            #Skip format
            continue
        widgets = list(iterate_format(video_format))
        formats.append("video/" + format_name)
        if (len(widgets) > 0):
            format_widgets["video/"+ format_name] = widgets
    return formats, format_widgets

def apply_format_widgets(format_name, kwargs):
    if os.path.exists(os.path.join(base_formats_dir, format_name + ".json")):
        video_format_path = os.path.join(base_formats_dir, format_name + ".json")
    else:
        video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name)
    with open(video_format_path, 'r') as stream:
        video_format = json.load(stream)
    for w in iterate_format(video_format):
        if w[0] not in kwargs:
            if len(w) > 2 and 'default' in w[2]:
                default = w[2]['default']
            else:
                if type(w[1]) is list:
                    default = w[1][0]
                else:
                    #NOTE: This doesn't respect max/min, but should be good enough as a fallback to a fallback to a fallback
                    default = {"BOOLEAN": False, "INT": 0, "FLOAT": 0, "STRING": ""}[w[1]]
            kwargs[w[0]] = default
            logger.warn(f"Missing input for {w[0][0]} has been set to {default}")
    wit = iterate_format(video_format, False)
    for w in wit:
        while isinstance(w, list):
            if len(w) == 1:
                #TODO: mapping=kwargs should be safer, but results in key errors, investigate why
                w = [Template(x).substitute(**kwargs) for x in w[0]]
                break
            elif isinstance(w[1], dict):
                w = w[1][str(kwargs[w[0]])]
            elif len(w) > 3:
                w = Template(w[3]).substitute(val=kwargs[w[0]])
            else:
                w = str(kwargs[w[0]])
        wit.send(w)
    return video_format

def tensor_to_int(tensor, bits):
    #TODO: investigate benefit of rounding by adding 0.5 before clip/cast
    tensor = tensor.cpu().numpy() * (2**bits-1)
    return np.clip(tensor, 0, (2**bits-1))
def tensor_to_shorts(tensor):
    return tensor_to_int(tensor, 16).astype(np.uint16)
def tensor_to_bytes(tensor):
    return tensor_to_int(tensor, 8).astype(np.uint8)

def ffmpeg_process(args, video_format, video_metadata, file_path, env):

    res = None
    frame_data = yield
    total_frames_output = 0
    if video_format.get('save_metadata', 'False') != 'False':
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
        metadata = json.dumps(video_metadata)
        metadata_path = os.path.join(folder_paths.get_temp_directory(), "metadata.txt")
        #metadata from file should  escape = ; # \ and newline
        metadata = metadata.replace("\\","\\\\")
        metadata = metadata.replace(";","\\;")
        metadata = metadata.replace("#","\\#")
        metadata = metadata.replace("=","\\=")
        metadata = metadata.replace("\n","\\\n")
        metadata = "comment=" + metadata
        with open(metadata_path, "w") as f:
            f.write(";FFMETADATA1\n")
            f.write(metadata)
        m_args = args[:1] + ["-i", metadata_path] + args[1:] + ["-metadata", "creation_time=now"]
        with subprocess.Popen(m_args + [file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    #TODO: skip flush for increased speed
                    frame_data = yield
                    total_frames_output+=1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                err = proc.stderr.read()
                #Check if output file exists. If it does, the re-execution
                #will also fail. This obscures the cause of the error
                #and seems to never occur concurrent to the metadata issue
                if os.path.exists(file_path):
                    raise Exception("An error occurred in the ffmpeg subprocess:\n" \
                            + err.decode(*ENCODE_ARGS))
                #Res was not set
                print(err.decode(*ENCODE_ARGS), end="", file=sys.stderr)
                logger.warn("An error occurred when saving with metadata")
    if res != b'':
        with subprocess.Popen(args + [file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    frame_data = yield
                    total_frames_output+=1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                res = proc.stderr.read()
                raise Exception("An error occurred in the ffmpeg subprocess:\n" \
                        + res.decode(*ENCODE_ARGS))
    yield total_frames_output
    if len(res) > 0:
        print(res.decode(*ENCODE_ARGS), end="", file=sys.stderr)

def gifski_process(args, dimensions, video_format, file_path, env):
    frame_data = yield
    with subprocess.Popen(args + video_format['main_pass'] + ['-f', 'yuv4mpegpipe', '-'],
                          stderr=subprocess.PIPE, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, env=env) as procff:
        with subprocess.Popen([gifski_path] + video_format['gifski_pass']
                              + ['-W', f'{dimensions[0]}', '-H', f'{dimensions[1]}']
                              + ['-q', '-o', file_path, '-'], stderr=subprocess.PIPE,
                              stdin=procff.stdout, stdout=subprocess.PIPE,
                              env=env) as procgs:
            try:
                while frame_data is not None:
                    procff.stdin.write(frame_data)
                    frame_data = yield
                procff.stdin.flush()
                procff.stdin.close()
                resff = procff.stderr.read()
                resgs = procgs.stderr.read()
                outgs = procgs.stdout.read()
            except BrokenPipeError as e:
                procff.stdin.close()
                resff = procff.stderr.read()
                resgs = procgs.stderr.read()
                raise Exception("An error occurred while creating gifski output\n" \
                        + "Make sure you are using gifski --version >=1.32.0\nffmpeg: " \
                        + resff.decode(*ENCODE_ARGS) + '\ngifski: ' + resgs.decode(*ENCODE_ARGS))
    if len(resff) > 0:
        print(resff.decode(*ENCODE_ARGS), end="", file=sys.stderr)
    if len(resgs) > 0:
        print(resgs.decode(*ENCODE_ARGS), end="", file=sys.stderr)
    #should always be empty as the quiet flag is passed
    if len(outgs) > 0:
        print(outgs.decode(*ENCODE_ARGS))

def to_pingpong(inp):
    if not hasattr(inp, "__getitem__"):
        inp = list(inp)
    yield from inp
    for i in range(len(inp)-2,0,-1):
        yield inp[i]

class VideoCombine:
    @classmethod
    def INPUT_TYPES(s):
        ffmpeg_formats, format_widgets = get_video_formats()
        format_widgets["image/webp"] = [['lossless', "BOOLEAN", {'default': True}]]
        return {
            "required": {
                "images": (imageOrLatent,),
                "frame_rate": (
                    floatOrInt,
                    {"default": 8, "min": 1, "step": 1},
                ),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "format": (["image/gif", "image/webp"] + ffmpeg_formats, {'formats': format_widgets}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
            },
            "hidden": ContainsAll({
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            }),
        }

    RETURN_TYPES = ("VHS_FILENAMES", "STRING",)
    RETURN_NAMES = ("Filenames", "files",)
    OUTPUT_NODE = True
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "combine_video"

    def combine_video(
        self,
        frame_rate: int,          # 视频帧率
        loop_count: int,          # 循环次数，0表示无限循环
        images=None,              # 输入图像序列
        latents=None,             # 潜在空间表示（可选）
        filename_prefix="AnimateDiff",  # 输出文件名前缀
        format="image/gif",       # 输出格式（image/gif, video/mp4等）
        pingpong=False,           # 是否启用乒乓效果（正向+反向播放）
        save_output=True,         # 是否保存到输出目录
        prompt=None,              # 提示词信息
        extra_pnginfo=None,       # 额外的PNG元数据信息
        audio=None,               # 音频数据（可选）
        unique_id=None,           # 唯一标识符
        manual_format_widgets=None,  # 手动格式控件（已弃用）
        meta_batch=None,          # 批处理元数据
        vae=None,                 # VAE模型（用于解码潜在表示）
        **kwargs                  # 其他关键字参数
    ):
        """
        将图像序列合成为视频或动画文件
        
        该方法支持多种输出格式，包括GIF动画和各种视频格式（通过FFmpeg）。
        可以处理普通图像或通过VAE解码的潜在表示。
        
        Returns:
            dict: 包含UI预览信息和结果文件路径的字典
        """
        # 输入数据预处理
        if latents is not None:
            images = latents  # 如果提供了潜在表示，使用它作为图像数据
        if images is None:
            return ((save_output, []),)  # 没有输入图像时直接返回
        if vae is not None:
            if isinstance(images, dict):
                images = images['samples']  # 从字典中提取样本数据
            else:
                vae = None  # 如果图像不是字典格式，则不使用VAE

        # 检查是否为空的张量
        if isinstance(images, torch.Tensor) and images.size(0) == 0:
            return ((save_output, []),)
        # 初始化处理参数
        num_frames = len(images)  # 获取总帧数
        pbar = ProgressBar(num_frames)  # 创建进度条
        
        # VAE解码处理（如果需要）
        if vae is not None:
            # 计算下采样比例和批处理大小
            downscale_ratio = getattr(vae, "downscale_ratio", 8)
            width = images.size(-1)*downscale_ratio
            height = images.size(-2)*downscale_ratio
            # 根据分辨率计算每批处理的帧数，避免内存溢出
            frames_per_batch = (1920 * 1080 * 16) // (width * height) or 1
            
            # Python 3.12添加了itertools.batched，这里为了兼容性自己实现
            def batched(it, n):
                """将迭代器分批处理"""
                while batch := tuple(itertools.islice(it, n)):
                    yield batch
                    
            def batched_encode(images, vae, frames_per_batch):
                """批量VAE解码"""
                for batch in batched(iter(images), frames_per_batch):
                    image_batch = torch.from_numpy(np.array(batch))
                    yield from vae.decode(image_batch)
                    
            # 执行批量解码
            images = batched_encode(images, vae, frames_per_batch)
            first_image = next(images)
            # 将第一张图像重新放回迭代器开头
            images = itertools.chain([first_image], images)
            # 确保图像是3维的（高度、宽度、通道），丢弃更高维度
            while len(first_image.shape) > 3:
                first_image = first_image[0]
        else:
            # 直接使用输入图像
            first_image = images[0]
            images = iter(images)
        # 获取输出路径信息
        output_dir = (
            folder_paths.get_output_directory()  # 保存到输出目录
            if save_output
            else folder_paths.get_temp_directory()  # 或临时目录
        )
        # 解析保存路径的各个组成部分
        (
            full_output_folder,  # 完整输出文件夹路径
            filename,            # 文件名
            _,                   # 未使用的返回值
            subfolder,           # 子文件夹
            _,                   # 未使用的返回值
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)
        output_files = []  # 存储所有输出文件路径

        # 准备元数据信息
        metadata = PngInfo()  # PNG元数据对象
        video_metadata = {}   # 视频元数据字典
        
        # 添加提示词信息到元数据
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
            video_metadata["prompt"] = json.dumps(prompt)
            
        # 添加额外的PNG信息到元数据
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                video_metadata[x] = extra_pnginfo[x]
            # 提取工作流的额外选项
            extra_options = extra_pnginfo.get('workflow', {}).get('extra', {})
        else:
            extra_options = {}
            
        # 添加创建时间戳
        metadata.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])

        # 处理文件计数器和批处理
        if meta_batch is not None and unique_id in meta_batch.outputs:
            # 从批处理中获取现有的计数器和输出进程
            (counter, output_process) = meta_batch.outputs[unique_id]
        else:
            # ComfyUI计数器解决方案：查找现有文件的最大编号
            max_counter = 0

            # 遍历现有文件以找到最大计数器值
            matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
            for existing_file in os.listdir(full_output_folder):
                # 检查文件是否匹配预期格式
                match = matcher.fullmatch(existing_file)
                if match:
                    # 提取文件名中的数字部分
                    file_counter = int(match.group(1))
                    # 更新最大计数器值
                    if file_counter > max_counter:
                        max_counter = file_counter

            # 计数器加1以获取下一个可用值
            counter = max_counter + 1
            output_process = None

        # 保存第一帧为PNG格式以保留元数据
        first_image_file = f"{filename}_{counter:05}.png"
        file_path = os.path.join(full_output_folder, first_image_file)
        # 检查是否需要保存元数据图像（默认为True）
        if extra_options.get('VHS_MetadataImage', True) != False:
            Image.fromarray(tensor_to_bytes(first_image)).save(
                file_path,
                pnginfo=metadata,  # 包含元数据
                compress_level=4,  # PNG压缩级别
            )
        output_files.append(file_path)  # 添加到输出文件列表

        # 解析输出格式
        format_type, format_ext = format.split("/")  # 分离格式类型和扩展名
        
        if format_type == "image":  # 图像格式处理（如GIF、WebP等）
            # 检查批处理兼容性
            if meta_batch is not None:
                raise Exception("Pillow('image/') formats are not compatible with batched output")
                
            # 根据格式设置特定参数
            image_kwargs = {}
            if format_ext == "gif":
                image_kwargs['disposal'] = 2  # GIF帧处理方式
            if format_ext == "webp":
                # 保存时间戳信息到EXIF
                exif = Image.Exif()
                exif[ExifTags.IFD.Exif] = {36867: datetime.datetime.now().isoformat(" ")[:19]}
                image_kwargs['exif'] = exif
                image_kwargs['lossless'] = kwargs.get("lossless", True)  # WebP无损压缩
            # 生成输出文件路径
            file = f"{filename}_{counter:05}.{format_ext}"
            file_path = os.path.join(full_output_folder, file)
            
            # 应用乒乓效果（如果启用）
            if pingpong:
                images = to_pingpong(images)  # 添加反向播放帧
                
            # 创建帧生成器，同时更新进度条
            def frames_gen(images):
                for i in images:
                    pbar.update(1)  # 更新进度
                    yield Image.fromarray(tensor_to_bytes(i))  # 转换为PIL图像
                    
            frames = frames_gen(images)
            
            # 使用Pillow直接保存动画图像
            next(frames).save(
                file_path,
                format=format_ext.upper(),           # 格式名称大写
                save_all=True,                       # 保存所有帧
                append_images=frames,                 # 追加其余帧
                duration=round(1000 / frame_rate),    # 每帧持续时间（毫秒）
                loop=loop_count,                      # 循环次数
                compress_level=4,                     # 压缩级别
                **image_kwargs                        # 格式特定参数
            )
            output_files.append(file_path)  # 添加到输出文件列表
        else:  # 视频格式处理（使用FFmpeg）
            # 检查FFmpeg是否可用
            if ffmpeg_path is None:
                raise ProcessLookupError(f"ffmpeg is required for video outputs and could not be found.\nIn order to use video outputs, you must either:\n- Install imageio-ffmpeg with pip,\n- Place a ffmpeg executable in {os.path.abspath('')}, or\n- Install ffmpeg and add it to the system path.")

            # 处理已弃用的手动格式控件参数
            if manual_format_widgets is not None:
                logger.warn("Format args can now be passed directly. The manual_format_widgets argument is now deprecated")
                kwargs.update(manual_format_widgets)

            # 检测图像是否包含Alpha通道
            has_alpha = first_image.shape[-1] == 4
            kwargs["has_alpha"] = has_alpha
            
            # 应用视频格式配置
            video_format = apply_format_widgets(format_ext, kwargs)
            dim_alignment = video_format.get("dim_alignment", 2)  # 尺寸对齐要求
            # 检查并处理尺寸对齐要求
            if (first_image.shape[1] % dim_alignment) or (first_image.shape[0] % dim_alignment):
                # 输出帧必须进行填充以满足对齐要求
                to_pad = (-first_image.shape[1] % dim_alignment,
                          -first_image.shape[0] % dim_alignment)
                # 计算四边填充值（左、右、上、下）
                padding = (to_pad[0]//2, to_pad[0] - to_pad[0]//2,
                           to_pad[1]//2, to_pad[1] - to_pad[1]//2)
                padfunc = torch.nn.ReplicationPad2d(padding)  # 复制填充函数
                
                def pad(image):
                    """对单张图像进行填充"""
                    image = image.permute((2,0,1))  # HWC转换为CHW格式
                    padded = padfunc(image.to(dtype=torch.float32))
                    return padded.permute((1,2,0))  # 转换回HWC格式
                    
                images = map(pad, images)  # 对所有图像应用填充
                # 计算填充后的尺寸
                dimensions = (-first_image.shape[1] % dim_alignment + first_image.shape[1],
                              -first_image.shape[0] % dim_alignment + first_image.shape[0])
                logger.warn("Output images were not of valid resolution and have had padding applied")
            else:
                # 尺寸已经对齐，直接使用原始尺寸
                dimensions = (first_image.shape[1], first_image.shape[0])
            # 设置循环参数
            if loop_count > 0:
                # 构建FFmpeg循环滤镜参数
                loop_args = ["-vf", "loop=loop=" + str(loop_count)+":size=" + str(num_frames)]
            else:
                loop_args = []  # 不循环
                
            # 处理乒乓效果
            if pingpong:
                if meta_batch is not None:
                    logger.error("pingpong is incompatible with batched output")
                images = to_pingpong(images)  # 添加反向播放帧
            # 根据颜色深度设置像素格式
            if video_format.get('input_color_depth', '8bit') == '16bit':
                # 16位颜色深度
                images = map(tensor_to_shorts, images)  # 转换为16位数据
                if has_alpha:
                    i_pix_fmt = 'rgba64'  # 16位RGBA
                else:
                    i_pix_fmt = 'rgb48'   # 16位RGB
            else:
                # 8位颜色深度（默认）
                images = map(tensor_to_bytes, images)  # 转换为8位数据
                if has_alpha:
                    i_pix_fmt = 'rgba'    # 8位RGBA
                else:
                    i_pix_fmt = 'rgb24'   # 8位RGB
            # 生成视频文件路径
            file = f"{filename}_{counter:05}.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)
            
            # 设置比特率参数
            bitrate_arg = []
            bitrate = video_format.get('bitrate')
            if bitrate is not None:
                # 根据配置决定使用Mbps还是Kbps
                bitrate_arg = ["-b:v", str(bitrate) + "M" if video_format.get('megabit') == 'True' else str(bitrate) + "K"]
            # 构建FFmpeg命令参数
            args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", i_pix_fmt,
                    # 颜色空间处理说明：
                    # 图像数据处于未定义的通用RGB颜色空间，实际上是sRGB。
                    # sRGB与BT.709有相同的色域和矩阵，但传输函数（伽马）不同，
                    # sRGB标准名称为IEC 61966-2-1。然而，YouTube等视频平台
                    # 标准化为完整的BT.709并相应转换颜色。这种最后时刻的颜色
                    # 变化可能会让用户困惑。我们可以通过在每种格式基础上
                    # "欺骗"传输函数来解决这个问题，即对于视频，我们告诉FFmpeg
                    # 它已经是BT.709了。另外，因为输入数据是RGB（不是YUV），
                    # 指定输入颜色空间为RGB更高效（减少缩放滤镜调用），
                    # 然后如果格式实际需要YUV，通过FFmpeg的-vf "scale=out_color_matrix=bt709"转换。
                    "-color_range", "pc",           # 颜色范围：PC（全范围）
                    "-colorspace", "rgb",           # 颜色空间：RGB
                    "-color_primaries", "bt709",    # 色域：BT.709
                    "-color_trc", video_format.get("fake_trc", "iec61966-2-1"),  # 传输函数
                    "-s", f"{dimensions[0]}x{dimensions[1]}",  # 视频尺寸
                    "-r", str(frame_rate),          # 帧率
                    "-i", "-"] \
                    + loop_args  # 添加循环参数

            # 将图像数据转换为字节流
            images = map(lambda x: x.tobytes(), images)
            
            # 设置环境变量
            env=os.environ.copy()  # 复制当前环境变量
            if "environment" in video_format:
                env.update(video_format["environment"])  # 添加格式特定的环境变量

            # 处理预处理阶段（如果需要）
            if "pre_pass" in video_format:
                if meta_batch is not None:
                    # 执行预处理需要保持对所有帧的访问。
                    # 潜在解决方案包括仅在内存中保留输出帧或使用带中间文件的3次处理，
                    # 但不应鼓励非常长的GIF
                    raise Exception("Formats which require a pre_pass are incompatible with Batch Manager.")
                    
                # 将所有图像数据合并为单个字节流
                images = [b''.join(images)]
                os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
                
                # 构建预处理命令参数
                in_args_len = args.index("-i") + 2  # "-i"和"-"之后的索引
                pre_pass_args = args[:in_args_len] + video_format['pre_pass']
                merge_filter_args(pre_pass_args)  # 合并滤镜参数
                
                # 执行预处理
                try:
                    subprocess.run(pre_pass_args, input=images[0], env=env,
                                   capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occurred in the ffmpeg prepass:\n" \
                            + e.stderr.decode(*ENCODE_ARGS))
            # 添加主处理阶段的输入参数（如果需要）
            if "inputs_main_pass" in video_format:
                in_args_len = args.index("-i") + 2  # "-i"和"-"之后的索引
                args = args[:in_args_len] + video_format['inputs_main_pass'] + args[in_args_len:]

            # 初始化输出处理进程
            if output_process is None:
                if 'gifski_pass' in video_format:
                    # 使用Gifski处理GIF格式
                    format = 'image/gif'
                    output_process = gifski_process(args, dimensions, video_format, file_path, env)
                else:
                    # 使用FFmpeg处理其他视频格式
                    args += video_format['main_pass'] + bitrate_arg
                    merge_filter_args(args)  # 合并滤镜参数
                    output_process = ffmpeg_process(args, video_format, video_metadata, file_path, env)
                    
                # 启动处理进程
                output_process.send(None)
                
                # 如果是批处理，保存进程信息
                if meta_batch is not None:
                    meta_batch.outputs[unique_id] = (counter, output_process)

            # 逐帧发送图像数据到处理进程
            for image in images:
                pbar.update(1)  # 更新进度条
                output_process.send(image)  # 发送图像数据
                
            # 处理批处理工作流
            if meta_batch is not None:
                requeue_workflow((meta_batch.unique_id, not meta_batch.has_closed_inputs))
                
            # 完成处理或继续批处理
            if meta_batch is None or meta_batch.has_closed_inputs:
                # 关闭管道并等待终止
                try:
                    total_frames_output = output_process.send(None)  # 获取输出帧数
                    output_process.send(None)  # 最终关闭信号
                except StopIteration:
                    pass
                    
                # 清理批处理状态
                if meta_batch is not None:
                    meta_batch.outputs.pop(unique_id)
                    if len(meta_batch.outputs) == 0:
                        meta_batch.reset()
            else:
                # 批处理未完成
                # TODO: 检查空输出是否会破坏其他自定义节点
                return {"ui": {"unfinished_batch": [True]}, "result": ((save_output, []),)}

            output_files.append(file_path)  # 添加视频文件到输出列表

            # 处理音频（如果提供）
            a_waveform = None
            if audio is not None:
                try:
                    # 安全检查VHS_LoadVideo产生的音频是否实际存在
                    a_waveform = audio['waveform']
                except:
                    pass
                    
            if a_waveform is not None:
                # 如果提供了音频输入，创建带音频的文件
                output_file_with_audio = f"{filename}_{counter:05}-audio.{video_format['extension']}"
                output_file_with_audio_path = os.path.join(full_output_folder, output_file_with_audio)
                
                # 检查视频格式是否支持音频
                if "audio_pass" not in video_format:
                    logger.warn("Selected video format does not have explicit audio support")
                    video_format["audio_pass"] = ["-c:a", "libopus"]  # 默认使用Opus编码


                # 构建带音频重编码的FFmpeg命令
                # TODO: 如果格式控件支持，暴露音频质量选项
                # TODO: 重新考虑强制apad/shortest
                channels = audio['waveform'].size(1)  # 音频通道数
                min_audio_dur = total_frames_output / frame_rate + 1  # 最小音频时长
                
                # 设置音频填充参数
                if video_format.get('trim_to_audio', 'False') != 'False':
                    apad = []  # 不填充，裁剪到音频长度
                else:
                    apad = ["-af", "apad=whole_dur="+str(min_audio_dur)]  # 填充音频到视频长度
                    
                # 构建音频混合命令
                mux_args = [ffmpeg_path, "-v", "error", "-n", "-i", file_path,  # 输入视频文件
                            "-ar", str(audio['sample_rate']),  # 音频采样率
                            "-ac", str(channels),              # 音频通道数
                            "-f", "f32le", "-i", "-",          # 音频输入格式
                            "-c:v", "copy"] \
                            + video_format["audio_pass"] \
                            + apad + ["-shortest", output_file_with_audio_path]  # 输出文件

                # 准备音频数据
                audio_data = audio['waveform'].squeeze(0).transpose(0,1) \
                        .numpy().tobytes()  # 转换音频数据为字节流
                        
                merge_filter_args(mux_args, '-af')  # 合并音频滤镜参数
                
                # 执行音频混合
                try:
                    res = subprocess.run(mux_args, input=audio_data,
                                         env=env, capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occured in the ffmpeg subprocess:\n" \
                            + e.stderr.decode(*ENCODE_ARGS))
                            
                # 输出错误信息（如果有）
                if res.stderr:
                    print(res.stderr.decode(*ENCODE_ARGS), end="", file=sys.stderr)
                    
                output_files.append(output_file_with_audio_path)  # 添加音频文件到输出列表
                
                # 返回带音频的文件给WebUI
                # 除非右键打开或保存，否则会被静音
                file = output_file_with_audio
        # 清理中间文件（如果配置为不保留）
        if extra_options.get('VHS_KeepIntermediate', True) == False:
            for intermediate in output_files[1:-1]:  # 保留第一个和最后一个文件
                if os.path.exists(intermediate):
                    os.remove(intermediate)
                    
        # 构建预览信息
        preview = {
                "filename": file,                                    # 文件名
                "subfolder": subfolder,                             # 子文件夹
                "type": "output" if save_output else "temp",        # 文件类型
                "format": format,                                   # 格式
                "frame_rate": frame_rate,                           # 帧率
                "workflow": first_image_file,                       # 工作流文件
                "fullpath": output_files[-1],                      # 完整路径
            }
            
        # 处理单帧PNG的特殊情况
        if num_frames == 1 and 'png' in format and '%03d' in file:
            preview['format'] = 'image/png'
            preview['filename'] = file.replace('%03d', '001')
            
        # 返回结果
        return {"ui": {"gifs": [preview]}, "result": ((save_output, output_files, [preview['fullpath']]),)}

class LoadAudio:
    @classmethod
    def INPUT_TYPES(s):
        #Hide ffmpeg formats if ffmpeg isn't available
        return {
            "required": {
                "audio_file": ("STRING", {"default": "input/", "vhs_path_extensions": ['wav','mp3','ogg','m4a','flac']}),
                },
            "optional" : {"seek_seconds": ("FLOAT", {"default": 0, "min": 0}),
                          "duration": ("FLOAT" , {"default": 0, "min": 0, "max": 10000000, "step": 0.01}),
                          }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/audio"
    FUNCTION = "load_audio"
    def load_audio(self, audio_file, seek_seconds, duration):
        audio_file = strip_path(audio_file)
        if audio_file is None or validate_path(audio_file) != True:
            raise Exception("audio_file is not a valid path: " + audio_file)
        if is_url(audio_file):
            audio_file = try_download_video(audio_file) or audio_file
        #Eagerly fetch the audio since the user must be using it if the
        #node executes, unlike Load Video
        return (get_audio(audio_file, start_time=seek_seconds, duration=duration),)

    @classmethod
    def IS_CHANGED(s, audio_file, seek_seconds):
        return hash_path(audio_file)

    @classmethod
    def VALIDATE_INPUTS(s, audio_file, **kwargs):
        return validate_path(audio_file, allow_none=True)

class LoadAudioUpload:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in audio_extensions):
                    files.append(f)
        return {"required": {
                    "audio": (sorted(files),),
                    "start_time": ("FLOAT" , {"default": 0, "min": 0, "max": 10000000, "step": 0.01}),
                    "duration": ("FLOAT" , {"default": 0, "min": 0, "max": 10000000, "step": 0.01}),
                     },
                }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/audio"

    RETURN_TYPES = ("AUDIO", )
    RETURN_NAMES = ("audio",)
    FUNCTION = "load_audio"

    def load_audio(self, start_time, duration, **kwargs):
        audio_file = folder_paths.get_annotated_filepath(strip_path(kwargs['audio']))
        if audio_file is None or validate_path(audio_file) != True:
            raise Exception("audio_file is not a valid path: " + audio_file)
        
        return (get_audio(audio_file, start_time, duration),)

    @classmethod
    def IS_CHANGED(s, audio, start_time, duration):
        audio_file = folder_paths.get_annotated_filepath(strip_path(audio))
        return hash_path(audio_file)

    @classmethod
    def VALIDATE_INPUTS(s, audio, **kwargs):
        audio_file = folder_paths.get_annotated_filepath(strip_path(audio))
        return validate_path(audio_file, allow_none=True)
class AudioToVHSAudio:
    """Legacy method for external nodes that utilized VHS_AUDIO,
    VHS_AUDIO is deprecated as a format and should no longer be used"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"audio": ("AUDIO",)}}
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/audio"

    RETURN_TYPES = ("VHS_AUDIO", )
    RETURN_NAMES = ("vhs_audio",)
    FUNCTION = "convert_audio"

    def convert_audio(self, audio):
        ar = str(audio['sample_rate'])
        ac = str(audio['waveform'].size(1))
        mux_args = [ffmpeg_path, "-f", "f32le", "-ar", ar, "-ac", ac,
                    "-i", "-", "-f", "wav", "-"]

        audio_data = audio['waveform'].squeeze(0).transpose(0,1) \
                .numpy().tobytes()
        try:
            res = subprocess.run(mux_args, input=audio_data,
                                 capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            raise Exception("An error occured in the ffmpeg subprocess:\n" \
                    + e.stderr.decode(*ENCODE_ARGS))
        if res.stderr:
            print(res.stderr.decode(*ENCODE_ARGS), end="", file=sys.stderr)
        return (lambda: res.stdout,)

class VHSAudioToAudio:
    """Legacy method for external nodes that utilized VHS_AUDIO,
    VHS_AUDIO is deprecated as a format and should no longer be used"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"vhs_audio": ("VHS_AUDIO",)}}
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/audio"

    RETURN_TYPES = ("AUDIO", )
    RETURN_NAMES = ("audio",)
    FUNCTION = "convert_audio"

    def convert_audio(self, vhs_audio):
        if not vhs_audio or not vhs_audio():
            raise Exception("audio input is not valid")
        args = [ffmpeg_path, "-i", '-']
        try:
            res =  subprocess.run(args + ["-f", "f32le", "-"], input=vhs_audio(),
                                  capture_output=True, check=True)
            audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
        except subprocess.CalledProcessError as e:
            raise Exception("An error occured in the ffmpeg subprocess:\n" \
                    + e.stderr.decode(*ENCODE_ARGS))
        match = re.search(', (\\d+) Hz, (\\w+), ',res.stderr.decode(*ENCODE_ARGS))
        if match:
            ar = int(match.group(1))
            #NOTE: Just throwing an error for other channel types right now
            #Will deal with issues if they come
            ac = {"mono": 1, "stereo": 2}[match.group(2)]
        else:
            ar = 44100
            ac = 2
        audio = audio.reshape((-1,ac)).transpose(0,1).unsqueeze(0)
        return ({'waveform': audio, 'sample_rate': ar},)

class PruneOutputs:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "filenames": ("VHS_FILENAMES",),
                    "options": (["Intermediate", "Intermediate and Utility"],)
                    }
                }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "prune_outputs"

    def prune_outputs(self, filenames, options):
        if len(filenames[1]) == 0:
            return ()
        assert(len(filenames[1]) <= 3 and len(filenames[1]) >= 2)
        delete_list = []
        if options in ["Intermediate", "Intermediate and Utility", "All"]:
            delete_list += filenames[1][1:-1]
        if options in ["Intermediate and Utility", "All"]:
            delete_list.append(filenames[1][0])
        if options in ["All"]:
            delete_list.append(filenames[1][-1])

        output_dirs = [folder_paths.get_output_directory(),
                       folder_paths.get_temp_directory()]
        for file in delete_list:
            #Check that path is actually an output directory
            if (os.path.commonpath([output_dirs[0], file]) != output_dirs[0]) \
                    and (os.path.commonpath([output_dirs[1], file]) != output_dirs[1]):
                        raise Exception("Tried to prune output from invalid directory: " + file)
            if os.path.exists(file):
                os.remove(file)
        return ()

class BatchManager:
    def __init__(self, frames_per_batch=-1):
        self.frames_per_batch = frames_per_batch
        self.inputs = {}
        self.outputs = {}
        self.unique_id = None
        self.has_closed_inputs = False
        self.total_frames = float('inf')
    def reset(self):
        self.close_inputs()
        for key in self.outputs:
            if getattr(self.outputs[key][-1], "gi_suspended", False):
                try:
                    self.outputs[key][-1].send(None)
                except StopIteration:
                    pass
        self.__init__(self.frames_per_batch)
    def has_open_inputs(self):
        return len(self.inputs) > 0
    def close_inputs(self):
        for key in self.inputs:
            if getattr(self.inputs[key][-1], "gi_suspended", False):
                try:
                    self.inputs[key][-1].send(1)
                except StopIteration:
                    pass
        self.inputs = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "frames_per_batch": ("INT", {"default": 16, "min": 1, "max": BIGMAX, "step": 1})
                    },
                "hidden": {
                    "prompt": "PROMPT",
                    "unique_id": "UNIQUE_ID"
                },
                }

    RETURN_TYPES = ("VHS_BatchManager",)
    RETURN_NAMES = ("meta_batch",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "update_batch"

    def update_batch(self, frames_per_batch, prompt=None, unique_id=None):
        if unique_id is not None and prompt is not None:
            requeue = prompt[unique_id]['inputs'].get('requeue', 0)
        else:
            requeue = 0
        if requeue == 0:
            self.reset()
            self.frames_per_batch = frames_per_batch
            self.unique_id = unique_id
        else:
            num_batches = (self.total_frames+self.frames_per_batch-1)//frames_per_batch
            print(f'Meta-Batch {requeue}/{num_batches}')
        #onExecuted seems to not be called unless some message is sent
        return (self,)


class VideoInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "video_info": ("VHS_VIDEOINFO",),
                    }
                }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = ("FLOAT","INT", "FLOAT", "INT", "INT", "FLOAT","INT", "FLOAT", "INT", "INT")
    RETURN_NAMES = (
        "source_fps🟨",
        "source_frame_count🟨",
        "source_duration🟨",
        "source_width🟨",
        "source_height🟨",
        "loaded_fps🟦",
        "loaded_frame_count🟦",
        "loaded_duration🟦",
        "loaded_width🟦",
        "loaded_height🟦",
    )
    FUNCTION = "get_video_info"

    def get_video_info(self, video_info):
        keys = ["fps", "frame_count", "duration", "width", "height"]
        
        source_info = []
        loaded_info = []

        for key in keys:
            source_info.append(video_info[f"source_{key}"])
            loaded_info.append(video_info[f"loaded_{key}"])

        return (*source_info, *loaded_info)


class VideoInfoSource:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "video_info": ("VHS_VIDEOINFO",),
                    }
                }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = ("FLOAT","INT", "FLOAT", "INT", "INT",)
    RETURN_NAMES = (
        "fps🟨",
        "frame_count🟨",
        "duration🟨",
        "width🟨",
        "height🟨",
    )
    FUNCTION = "get_video_info"

    def get_video_info(self, video_info):
        keys = ["fps", "frame_count", "duration", "width", "height"]
        
        source_info = []

        for key in keys:
            source_info.append(video_info[f"source_{key}"])

        return (*source_info,)


class VideoInfoLoaded:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "video_info": ("VHS_VIDEOINFO",),
                    }
                }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = ("FLOAT","INT", "FLOAT", "INT", "INT",)
    RETURN_NAMES = (
        "fps🟦",
        "frame_count🟦",
        "duration🟦",
        "width🟦",
        "height🟦",
    )
    FUNCTION = "get_video_info"

    def get_video_info(self, video_info):
        keys = ["fps", "frame_count", "duration", "width", "height"]
        
        loaded_info = []

        for key in keys:
            loaded_info.append(video_info[f"loaded_{key}"])

        return (*loaded_info,)

class SelectFilename:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"filenames": ("VHS_FILENAMES",), "index": ("INT", {"default": -1, "step": 1, "min": -1})}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES =("Filename",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "select_filename"

    def select_filename(self, filenames, index):
        return (filenames[1][index],)
class Unbatch:
    class Any(str):
        def __ne__(self, other):
            return False
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"batched": ("*",)}}
    RETURN_TYPES = (Any('*'),)
    INPUT_IS_LIST = True
    RETURN_NAMES =("unbatched",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "unbatch"
    def unbatch(self, batched):
        if isinstance(batched[0], torch.Tensor):
            return (torch.cat(batched),)
        if isinstance(batched[0], dict):
            out = batched[0].copy()
            if 'samples' in out:
                out['samples'] = torch.cat([x['samples'] for x in batched])
            if 'waveform' in out:
                out['waveform'] = torch.cat([x['waveform'] for x in batched])
            out.pop('batch_index', None)
            return (out,)
        return (functools.reduce(lambda x,y: x+y, batched),)
    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True
class SelectLatest:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"filename_prefix": ("STRING", {'default': 'output/AnimateDiff', 'vhs_path_extensions': []}),
                             "filename_postfix": ("STRING", {"placeholder": ".webm"})}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES =("Filename",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "select_latest"
    EXPERIMENTAL = True

    def select_latest(self, filename_prefix, filename_postfix):
        assert False, "Not Reachable"

NODE_CLASS_MAPPINGS = {
    "VHS_VideoCombine": VideoCombine,
    "VHS_LoadVideo": LoadVideoUpload,
    "VHS_LoadVideoPath": LoadVideoPath,
    "VHS_LoadVideoFFmpeg": LoadVideoFFmpegUpload,
    "VHS_LoadVideoFFmpegPath": LoadVideoFFmpegPath,
    "VHS_LoadImagePath": LoadImagePath,
    "VHS_LoadImages": LoadImagesFromDirectoryUpload,
    "VHS_LoadImagesPath": LoadImagesFromDirectoryPath,
    "VHS_LoadAudio": LoadAudio,
    "VHS_LoadAudioUpload": LoadAudioUpload,
    "VHS_AudioToVHSAudio": AudioToVHSAudio,
    "VHS_VHSAudioToAudio": VHSAudioToAudio,
    "VHS_PruneOutputs": PruneOutputs,
    "VHS_BatchManager": BatchManager,
    "VHS_VideoInfo": VideoInfo,
    "VHS_VideoInfoSource": VideoInfoSource,
    "VHS_VideoInfoLoaded": VideoInfoLoaded,
    "VHS_SelectFilename": SelectFilename,
    # Batched Nodes
    "VHS_VAEEncodeBatched": VAEEncodeBatched,
    "VHS_VAEDecodeBatched": VAEDecodeBatched,
    # Latent and Image nodes
    "VHS_SplitLatents": SplitLatents,
    "VHS_SplitImages": SplitImages,
    "VHS_SplitMasks": SplitMasks,
    "VHS_MergeLatents": MergeLatents,
    "VHS_MergeImages": MergeImages,
    "VHS_MergeMasks": MergeMasks,
    "VHS_GetLatentCount": GetLatentCount,
    "VHS_GetImageCount": GetImageCount,
    "VHS_GetMaskCount": GetMaskCount,
    "VHS_DuplicateLatents": RepeatLatents,
    "VHS_DuplicateImages": RepeatImages,
    "VHS_DuplicateMasks": RepeatMasks,
    "VHS_SelectEveryNthLatent": SelectEveryNthLatent,
    "VHS_SelectEveryNthImage": SelectEveryNthImage,
    "VHS_SelectEveryNthMask": SelectEveryNthMask,
    "VHS_SelectLatents": SelectLatents,
    "VHS_SelectImages": SelectImages,
    "VHS_SelectMasks": SelectMasks,
    "VHS_Unbatch": Unbatch,
    "VHS_SelectLatest": SelectLatest,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_VideoCombine": "Video Combine 🎥🅥🅗🅢",
    "VHS_LoadVideo": "Load Video (Upload) 🎥🅥🅗🅢",
    "VHS_LoadVideoPath": "Load Video (Path) 🎥🅥🅗🅢",
    "VHS_LoadVideoFFmpeg": "Load Video FFmpeg (Upload) 🎥🅥🅗🅢",
    "VHS_LoadVideoFFmpegPath": "Load Video FFmpeg (Path) 🎥🅥🅗🅢",
    "VHS_LoadImagePath": "Load Image (Path) 🎥🅥🅗🅢",
    "VHS_LoadImages": "Load Images (Upload) 🎥🅥🅗🅢",
    "VHS_LoadImagesPath": "Load Images (Path) 🎥🅥🅗🅢",
    "VHS_LoadAudio": "Load Audio (Path)🎥🅥🅗🅢",
    "VHS_LoadAudioUpload": "Load Audio (Upload)🎥🅥🅗🅢",
    "VHS_AudioToVHSAudio": "Audio to legacy VHS_AUDIO🎥🅥🅗🅢",
    "VHS_VHSAudioToAudio": "Legacy VHS_AUDIO to Audio🎥🅥🅗🅢",
    "VHS_PruneOutputs": "Prune Outputs 🎥🅥🅗🅢",
    "VHS_BatchManager": "Meta Batch Manager 🎥🅥🅗🅢",
    "VHS_VideoInfo": "Video Info 🎥🅥🅗🅢",
    "VHS_VideoInfoSource": "Video Info (Source) 🎥🅥🅗🅢",
    "VHS_VideoInfoLoaded": "Video Info (Loaded) 🎥🅥🅗🅢",
    "VHS_SelectFilename": "Select Filename 🎥🅥🅗🅢",
    # Batched Nodes
    "VHS_VAEEncodeBatched": "VAE Encode Batched 🎥🅥🅗🅢",
    "VHS_VAEDecodeBatched": "VAE Decode Batched 🎥🅥🅗🅢",
    # Latent and Image nodes
    "VHS_SplitLatents": "Split Latents 🎥🅥🅗🅢",
    "VHS_SplitImages": "Split Images 🎥🅥🅗🅢",
    "VHS_SplitMasks": "Split Masks 🎥🅥🅗🅢",
    "VHS_MergeLatents": "Merge Latents 🎥🅥🅗🅢",
    "VHS_MergeImages": "Merge Images 🎥🅥🅗🅢",
    "VHS_MergeMasks": "Merge Masks 🎥🅥🅗🅢",
    "VHS_GetLatentCount": "Get Latent Count 🎥🅥🅗🅢",
    "VHS_GetImageCount": "Get Image Count 🎥🅥🅗🅢",
    "VHS_GetMaskCount": "Get Mask Count 🎥🅥🅗🅢",
    "VHS_DuplicateLatents": "Repeat Latents 🎥🅥🅗🅢",
    "VHS_DuplicateImages": "Repeat Images 🎥🅥🅗🅢",
    "VHS_DuplicateMasks": "Repeat Masks 🎥🅥🅗🅢",
    "VHS_SelectEveryNthLatent": "Select Every Nth Latent 🎥🅥🅗🅢",
    "VHS_SelectEveryNthImage": "Select Every Nth Image 🎥🅥🅗🅢",
    "VHS_SelectEveryNthMask": "Select Every Nth Mask 🎥🅥🅗🅢",
    "VHS_SelectLatents": "Select Latents 🎥🅥🅗🅢",
    "VHS_SelectImages": "Select Images 🎥🅥🅗🅢",
    "VHS_SelectMasks": "Select Masks 🎥🅥🅗🅢",
    "VHS_Unbatch":  "Unbatch 🎥🅥🅗🅢",
    "VHS_SelectLatest": "Select Latest 🎥🅥🅗🅢",
}
