# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py
import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x


class Upsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        print(f"Upsample3D __init__ channels: {channels},
            out_channels: {out_channels},
            self.out_channels: {self.out_channels},
            use_conv: {use_conv}, 
            use_conv_transpose: {use_conv_transpose}, 
            name: {name}")

        conv = None
        if use_conv_transpose:
            raise NotImplementedError
        elif use_conv:
            conv = InflatedConv3d(self.channels, self.out_channels, 3, padding=1)

        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        if(output_size is not None):
            print(f"Upsample3D output_size shape: {output_size.shape}, dtype: {output_size.dtype}")


        if self.use_conv_transpose:
            raise NotImplementedError

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)
            print(f"Upsample3D hidden_states1 shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=[1.0, 2.0, 2.0], mode="nearest")
            print(f"Upsample3D forward hidden_states2 shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")
            print(f"Upsample3D forward hidden_states3 shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
                print(f"Upsample3D forward hidden_states4 shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
            else:
                hidden_states = self.Conv2d_0(hidden_states)
                print(f"Upsample3D forward hidden_states5 shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")

        return hidden_states


class Downsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        print(f"Downsample3D __init__ channels: {channels},
            out_channels: {out_channels},
            self.out_channels: {self.out_channels},
            use_conv: {use_conv}, 
            padding: {padding}, 
            stride: {stride}, 
            name: {name}")

        if use_conv:
            conv = InflatedConv3d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            raise NotImplementedError

        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            raise NotImplementedError

        assert hidden_states.shape[1] == self.channels
        print(f"Downsample3D forward hidden_states1 shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
        hidden_states = self.conv(hidden_states)
        print(f"Downsample3D forward hidden_states2 shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")

        return hidden_states


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        output_scale_factor=1.0,
        use_in_shortcut=None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        print(f"ResnetBlock3D __init__ in_channels: {in_channels},
            out_channels: {out_channels},
            self.out_channels: {self.out_channels},
            temb_channels: {temb_channels}, 
            dropout: {dropout}, 
            groups: {groups}, 
            groups_out: {groups_out}, 
            pre_norm: {pre_norm}, 
            eps: {eps}, 
            non_linearity: {non_linearity}, 
            time_embedding_norm: {time_embedding_norm}, 
            output_scale_factor: {output_scale_factor}, 
            use_in_shortcut: {use_in_shortcut}, 
            conv_shortcut: {conv_shortcut}, 
            output_scale_factor: {output_scale_factor}")

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = InflatedConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            self.time_emb_proj = torch.nn.Linear(temb_channels, time_emb_proj_out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = InflatedConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = InflatedConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, temb):
        
        print(f"ResnetBlock3D forward temb shape: {temb.shape}, dtype: {temb.dtype}") 

        hidden_states = input_tensor
        print(f"ResnetBlock3D forward hidden_states1 shape: {hidden_states.shape}, dtype: {hidden_states.dtype}") #shape: torch.Size([1, 320, 16, 20, 32]), dtype: torch.float16
        
        hidden_states = self.norm1(hidden_states)

        print(f"ResnetBlock3D forward  hidden_states2 shape: {hidden_states.shape}, dtype: {hidden_states.dtype}") #shape: torch.Size([1, 320, 16, 20, 32]), dtype: torch.float32

        hidden_states = self.nonlinearity(hidden_states)

        print(f"ResnetBlock3D forward  hidden_states3 shape: {hidden_states.shape}, dtype: {hidden_states.dtype}") #shape: torch.Size([1, 320, 16, 20, 32]), dtype: torch.float32

        hidden_states = self.conv1(hidden_states)

        print(f"ResnetBlock3D forward  hidden_states4 shape: {hidden_states.shape}, dtype: {hidden_states.dtype}") #shape: torch.Size([1, 640, 16, 20, 32]), dtype: torch.float16

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]
            print(f"ResnetBlock3D forward temb1 shape: {temb.shape}, dtype: {temb.dtype}") 


        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb
            print(f"ResnetBlock3D forward hidden_states5 shape: {hidden_states.shape}, dtype: {hidden_states.dtype}") #shape: torch.Size([1, 640, 16, 20, 32]), dtype: torch.float16

        hidden_states = self.norm2(hidden_states)

        print(f"ResnetBlock3D forward hidden_states6 shape: {hidden_states.shape}, dtype: {hidden_states.dtype}") #shape: torch.Size([1, 640, 16, 20, 32]), dtype: torch.float32

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift
            print(f"ResnetBlock3D forward hidden_states7 shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")

        hidden_states = self.nonlinearity(hidden_states)

        print(f"ResnetBlock3D forward hidden_states8 shape: {hidden_states.shape}, dtype: {hidden_states.dtype}") #shape: torch.Size([1, 640, 16, 20, 32]), dtype: torch.float32

        hidden_states = self.dropout(hidden_states)

        print(f"ResnetBlock3D forward hidden_states9 shape: {hidden_states.shape}, dtype: {hidden_states.dtype}") #shape: torch.Size([1, 640, 16, 20, 32]), dtype: torch.float32

        hidden_states = self.conv2(hidden_states)

        print(f"ResnetBlock3D forward hidden_states10 shape: {hidden_states.shape}, dtype: {hidden_states.dtype}") #shape: torch.Size([1, 640, 16, 20, 32]), dtype: torch.float16

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
            print(f"ResnetBlock3D forward input_tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}") #shape: torch.Size([1, 640, 16, 20, 32]), dtype: torch.float16

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class Mish(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))
    