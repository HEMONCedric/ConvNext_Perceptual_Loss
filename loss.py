import torchvision
import torch.nn as nn
import torch
from utils import gram_matrix


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, device : str) -> None:
        super().__init__()
        blocks = []
        self.device = device
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4])
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9])
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16])
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23])
        self.blocks = torch.nn.ModuleList(blocks).eval().to(device)
        self.blocks.requires_grad_(False)
        self.transform = torch.nn.functional.interpolate
        self.style_layers = [0,1,2,3]
        self.style_layers_weight = [1,1,1,1]
        self.feature_layers = [0,1,2]
        self.feature_layers_weight = [0,0,1]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.loss_func = torch.nn.MSELoss(reduction='mean')
        
    def prepare(self, input: torch.Tensor) -> torch.Tensor:
        input = input.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        return input
    
    def loss_eval(self, input: torch.Tensor, target_style: torch.Tensor, target_content: torch.Tensor):
        input = self.prepare(input)
        target_style = self.prepare(target_style)
        target_content = self.prepare(target_content)
        
        loss_content = 0.0
        loss_style = 0.0
        x = input
        y = target_style
        z = target_content
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.feature_layers:
                z = block(z)
                loss_content += self.loss_func(x, z)*self.feature_layers_weight[i]
                
            if i in self.style_layers:
                y = block(y)
                gram_x = gram_matrix(x.double())
                gram_y = gram_matrix(y.double())
                loss_style += self.loss_func(gram_x, gram_y)*self.style_layers_weight[i]
        return loss_content, loss_style

    def forward(self, input: torch.Tensor, target_style: torch.Tensor, target_content: torch.Tensor) -> torch.Tensor:
        loss_content = 0.0
        loss_style = 0.0
        if len(input.shape)==4:
            loss_content, loss_style = self.loss_eval(input, target_style, target_content)
            loss_content /= input.shape[0] 
            loss_style /= input.shape[0] 
        else:
            for i in range(input.shape[2]):
                loss_content_tmp, loss_style_tmp = self.loss_eval(input[:,0,i], target_style[:,0,i], target_content[:,0,i])
                loss_content += loss_content_tmp
                loss_style += loss_style_tmp
            loss_content /= input.shape[2]
            loss_style /= input.shape[2]
        loss_content *= 0.1
        loss_style *= 30
        return loss_content+loss_style

class ConvNextPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(ConvNextPerceptualLoss, self).__init__()
        blocks = []
        self.device = device
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        blocks.append(torchvision.models.convnext.convnext_tiny(weights=weights).features[:1])
        blocks.append(torchvision.models.convnext.convnext_tiny(weights=weights).features[1:3])
        blocks.append(torchvision.models.convnext.convnext_tiny(weights=weights).features[3:5])
        self.blocks = nn.ModuleList(blocks).eval().to(device=self.device)
        self.blocks.requires_grad_(False)
        self.transform = nn.functional.interpolate
        # Lorsqu'on a le pCT  [10,2,1]  ou  [2,0,1] lorsqu'on a seulement un atlas
        self.style_layers = [0,1,2]
        self.style_layers_weight = [10,2,1]  
        self.feature_layers = [0,1]
        self.feature_layers_weight = [2,1]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def prepare(self, input: torch.Tensor):
        if input.shape[0] >1: 
            input = input.repeat(1, 3, 1, 1)
        else:
            input = input.repeat(1, 3, 1, 1)
        input = (input+1)/2
        input = (input-self.mean) / self.std
        # input = self.transform(input, mode='bilinear', size=(320, 416), align_corners=False)
        return input
        
    def loss_eval(self, input : torch.Tensor, target_style, target_content):
        input = self.prepare(input)
        target_style = self.prepare(target_style)
        target_content = self.prepare(target_content)

        loss_content = 0.0
        loss_style = 0.0
        x = input
        y_content = target_content
        y_style = target_style
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.style_layers:
                y_style = block(y_style)
                gram_y = gram_matrix(y_style.float())
                gram_x = gram_matrix(x.float())
                loss_style +=  torch.nn.L1Loss(reduction='sum')(gram_x, gram_y)*self.style_layers_weight[i]
            if i in self.feature_layers:
                y_content = block(y_content)
                loss_content += torch.nn.L1Loss(reduction='mean')(x.float(), y_content.float())*self.feature_layers_weight[i]

        return loss_content, loss_style

    def forward(self, input, target_style, target_content):
        loss_content = 0.0
        loss_style = 0.0
        if len(input.shape)==4:
            loss_content, loss_style = self.loss_eval(input, target_style, target_content)
            loss_content /= input.shape[0] 
            loss_style /= input.shape[0] 
        else:
            for i in range(input.shape[2]):
                loss_content_tmp, loss_style_tmp = self.loss_eval(input[:,0,i], target_style[:,0,i], target_content[:,0,i])
                loss_content += loss_content_tmp
                loss_style += loss_style_tmp
            loss_content /= input.shape[2] 
            loss_style /= input.shape[2] 
        loss_content*= 50
        loss_style*= 20
        return loss_content+loss_style
    

    class ConvNextPerceptualLoss_Inference(nn.Module):
    def __init__(self, device):
        super(ConvNextPerceptualLoss_Inference, self).__init__()
        blocks = []
        self.device = device
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        blocks.append(torchvision.models.convnext.convnext_tiny(weights=weights).features[:1])
        blocks.append(torchvision.models.convnext.convnext_tiny(weights=weights).features[1:3])
        blocks.append(torchvision.models.convnext.convnext_tiny(weights=weights).features[3:5])
        self.blocks = nn.ModuleList(blocks).eval().to(device=self.device)
        self.blocks.requires_grad_(False)
        self.transform = nn.functional.interpolate
        # Lorsqu'on a le pCT  [10,2,1]  ou  [2,0,1] lorsqu'on a seulement un atlas
        self.style_layers_weight = [10,2,1]         
        self.content_layers_weight = [2,1,0] 
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device=self.device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device=self.device))
        self.layers_content: Dict[List[torch.Tensor]]= {}
        self.layers_style: Dict[List[torch.Tensor]] = {}

    def prepare(self, input: torch.Tensor):
        input = input.repeat(1, 3, 1, 1)
        input = (input+1)/2
        input = (input-self.mean) / self.std
        # input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        return input
        
    def loss_eval(self, input : torch.Tensor, target_style, target_content, new: bool, slice:int):
        input = self.prepare(input)
        loss_content = 0.0
        loss_style = 0.0
        x = input
        if new:
            y_style = self.prepare(target_style)
            y_content = self.prepare(target_content)

            for i, block in enumerate(self.blocks):
                y_style = block(y_style)
                gram_y = gram_matrix(y_style)
                if slice not in self.layers_style:
                    self.layers_style[slice] = []
                if slice not in self.layers_content:
                    self.layers_content[slice] = []
                self.layers_style[slice].append(gram_y.detach())
                if i<2:
                    y_content = block(y_content)
                    self.layers_content[slice].append(y_content.detach())
                else:
                    self.layers_content[slice].append(0)

        
        for i, block in enumerate(self.blocks):
            x = block(x)
            gram_x = gram_matrix(x)
            loss_style +=  self.style_layers_weight[i]*torch.nn.L1Loss(reduction='sum')(gram_x, self.layers_style[slice][i])            
            if i<2:
                loss_content += self.content_layers_weight[i]*torch.nn.L1Loss(reduction='mean')(x, self.layers_content[slice][i])
        return loss_content, loss_style

    def forward(self, input, target_style, target_content, new=False):
        loss_content = 0.0
        loss_style = 0.0
        for i in range(input.shape[2]):
            loss_content_tmp, loss_style_tmp = self.loss_eval(input[:,0,i], target_style[:,0,i], target_content[:,0,i], new, i)
            loss_content += loss_content_tmp
            loss_style += loss_style_tmp
        loss_content /= input.shape[2] 
        loss_style /= input.shape[2]
        loss_content*= 50
        loss_style*= 20  # 25 ou 20
        return loss_content+loss_style

    
