import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

SEED = 3
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# 输入support set（5，1024） reference set1（64，1024） reference set2（64，1024） query set（75，1024）
# 输出reweighting之后的分类器。

def latent_loss(text_encoder):
    # text_encoder(5,512)
    c_dim = list(text_encoder.size())[-1] # 512
    # split the context into mean and variance predicted by task context encoder
    z_dim = c_dim //2  # 256
    c_mu = text_encoder[:,:z_dim]  # 256
    c_log_var = text_encoder[:,z_dim: ] # 256
    z_mean = c_mu
    z_stddev = torch.exp(c_log_var/2)
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

def Task_concate(reference):
    r_way, r_shot, _ =  reference.size() #(64,2,1024)
    r_pair = []  
    for i in range(0,r_way):
        current_class = []
        for j in range(0,r_shot):
            for k in range(j+1,r_shot):
                pair_tempt = torch.cat((reference[i][j], reference[i][k]),0) 
                current_class.append(pair_tempt) #每次放入(shot,fc*2,fw,fh)
        current_class = torch.stack(current_class, 0)  
        r_pair.append(current_class)# 每次放入(shot,way-1,shot,fc*2,fw,fh)
    r_pair = torch.stack(r_pair, 0) #(way,shot,way-1,shot,fc*2,fw,fh)
    
    return r_pair


class GeneratorNet(nn.Module):
    
    def __init__(self, N_generate):
        super(GeneratorNet, self).__init__()
        self.N_generate = N_generate
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.encoder1 = Encoder1()
        self.encoder2 = Encoder2()
        self.sampler1 = Sampler1(self.N_generate)
        self.sampler2 = Sampler2()
        self.decoder = Decoder(self.N_generate)
        
    def forward(self, support_set, reference):
        delta = self.encoder1(reference)
        diversity = self.sampler1(delta) # [64,1024]
        reweighting = self.encoder2(support_set) # (5,512)
        kl_loss = latent_loss(reweighting)
        reweighting = self.sampler2(reweighting) # (5,256)
        reweighting = self.decoder(reweighting) # (5,65)
        reweighting = torch.unsqueeze(reweighting, 1)#(5,1,65)
        support_set = torch.unsqueeze(support_set, 1)#(5,1,1024)
        prototype = torch.zeros(5,1024).cuda()
        support_all = torch.zeros(5,self.N_generate+1,1024).cuda()
        for i in range(5):
            current_support_set = support_set[i]
            current_generate_set = diversity * current_support_set + current_support_set #(64,1024)*(1024)
            current_reweighting = reweighting[i] # (1,64)
            current_support_all = torch.cat((current_support_set, current_generate_set), 0) # (65,1024)
            support_all[i] = current_support_all
            # 下面进行reweighting操作。
            prototype[i] = torch.squeeze(torch.mm(current_reweighting, current_support_all))
        
        inter_class_diversity_loss = 0
        intra_class_diversity_loss = 0
        support_all = torch.unsqueeze(support_all, 2)
        prototype = torch.unsqueeze(prototype, 1)#(5,1,1024)
        for i in range(5):
            for j in range(i+1, 5):
                inter_class_diversity_loss += F.pairwise_distance(prototype[i], prototype[j], 2)
            for k in range(self.N_generate+1):  
                intra_class_diversity_loss += F.pairwise_distance(support_all[i][k], prototype[i], 2)  
        prototype = torch.squeeze(prototype)#(5,1024)    
        loss_diversity = inter_class_diversity_loss/intra_class_diversity_loss
      
        return prototype, loss_diversity, kl_loss
    
    

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        norm = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(norm)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

# 输入 reference（64，2,1024）
# 输出 编码向量（1，2048）
class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1, self).__init__()
        self.fc1 = nn.Linear(2048, 2048)
        self.leakyRelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 2048)
        self.leakyRelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout2 = nn.Dropout(0.5)
        
    def forward(self, reference):
        r_pair = Task_concate(reference) # (r_way, r_shot, 2048)
        r_way, r_shot, _ = r_pair.size()
        r_pair = r_pair.view(r_way*r_shot, -1) # (64*2,2048)

        # 对reference编码
        x = self.fc1(r_pair) 
        x = self.leakyRelu1(x)
        x = self.dropout1(x)
        x = self.fc2(x) 
        x = self.leakyRelu2(x)
        x = self.dropout2(x)
        x = x.view(r_way, r_shot, -1) # (64, 2,2048)

        x = torch.mean(x,[0,1]) #(2048)

        return x

# 输入 support set (5,1024)
# 输出 编码向量（5，512）
class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()
        self.fc = nn.Linear(1024, 512)
        self.leakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
       
        x = self.fc(x) 
        x = self.leakyRelu(x)
        x = self.dropout(x)
      
        return x

class Decoder(nn.Module):
    def __init__(self, N_generate):
        super(Decoder, self).__init__()
        self.N_generate = N_generate
        self.fc = nn.Linear(256, self.N_generate+1) # reweighting系数是通过decoder得到的，也就意味着生成样本的个数训练和测试要一致。也就是说测试的时候不能任意个数采样了。
        self.leakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
       
        x = self.fc(x) 
        x = self.leakyRelu(x)
        x = self.dropout(x)
        x = F.softmax(x, dim=1) # (5,64)
      
        return x

# 输入[2048]
# 输出[64，1024]
class Sampler1(nn.Module):
    def __init__(self, N_generate):
        super(Sampler1, self).__init__()
        self.N_generate = N_generate
       
    def forward(self, delta):

        z_dim = 1024
        c_mu = delta[:z_dim]
        c_log_var = delta[z_dim: ]
        z_signal = torch.randn(self.N_generate, z_dim).cuda() #(5,64)
        z_c = c_mu + torch.exp(c_log_var/2)*z_signal

        return z_c



# 输入[5,512]
# 输出[5,256]
class Sampler2(nn.Module):
    def __init__(self):
        super(Sampler2, self).__init__()
       
    def forward(self, x):

        z_dim = 256
        c_mu = x[:,:z_dim]
        c_log_var = x[:,z_dim: ]
        z_signal = torch.randn(5, z_dim).cuda() #(5,256)
        z_c = c_mu + torch.exp(c_log_var/2)*z_signal

        return z_c

if __name__ == '__main__':
    support_set = torch.rand(5,1024).cuda()
    reference = torch.rand(8,3,1024).cuda()
    generatorNet = GeneratorNet(32).cuda()
    gen_feature, loss, kl_loss = generatorNet(support_set, reference)
    print(gen_feature.size())









