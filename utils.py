import torch

def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram


class Resnetseg(torch.nn.Module):

    def __init__(self, input_nc=1, output_nc=1, dim=3, n_blocks=2):
        super().__init__()
        norm_layer = getTorchModule("BatchNorm", dim=dim)
        use_bias = False

        conv = getTorchModule("Conv", dim=dim)
        self.first_conv = torch.nn.Sequential(*[conv(input_nc, 16, kernel_size=3, padding=1, bias=use_bias), norm_layer(16), nn.ReLU(True)])
        self.downsampling = torch.nn.Sequential(*[conv(16, 16, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(16), nn.ReLU(True)])

        for i in range(n_blocks):       # add ResNet blocks
            exec('self.block'+str(i)+' = ResnetBlock2(16, dim=dim, use_bias=use_bias)') in locals(), globals()

        for i in range(8):
            exec("self.final_layers"+str(i)+'= torch.nn.Sequential(*[conv(16, 16, kernel_size=3, padding=1, bias=use_bias), norm_layer(16), nn.ReLU(True), torch.nn.Upsample(scale_factor=2), conv(16, 8, kernel_size=3, padding=1, bias=use_bias), norm_layer(8), nn.ReLU(True),conv(8, output_nc, kernel_size=1, padding=0, bias=use_bias)])') in locals(), globals()
    
    def forward(self, x : torch.Tensor, number : int) -> torch.Tensor:
        x = self.first_conv(x)
        x = self.downsampling(x)
        x = self.block0(x)
        x = self.block1(x)
        exec('self.out0 = self.final_layers'+str(number)+'(x)') in locals(), globals()
        return self.out0
    
    
class ResnetBlock2(nn.Module):
    """Define a Resnet block"""

    def __init__(self, nb_ch, dim, use_bias):
        super(ResnetBlock2, self).__init__()
        self.conv_block = self.build_conv_block(nb_ch, dim, use_bias)

    def build_conv_block(self, nb_ch, dim, use_bias):
        conv_block = []
        conv = getTorchModule("Conv", dim=dim)
        conv_block += [conv(nb_ch, nb_ch, kernel_size=3, padding=1, bias=use_bias), nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x) 
        return out
