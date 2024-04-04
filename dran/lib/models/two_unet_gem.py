""" Modified from PixLoc https://psarlin.com/pixloc/ """

from symbol import decorated
import torchvision
import torch
import torch.nn as nn
from collections import OrderedDict

from pathlib import Path

from .base_model import BaseModel
from .utils import checkpointed

from ...retrieval.cirtorch.layers.pooling import GeM
from ...retrieval.cirtorch.layers.normalization import L2N

import time

class DecoderBlock(nn.Module):
    def __init__(self, previous, skip, out, num_convs=1, norm=nn.BatchNorm2d):
        super().__init__()

        self.upsample = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False)

        layers = []
        for i in range(num_convs):
            conv = nn.Conv2d(
                previous+skip if i == 0 else out, out,
                kernel_size=3, padding=1, bias=norm is None)
            layers.append(conv)
            if norm is not None:
                layers.append(norm(out))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, previous, skip):
        upsampled = self.upsample(previous) #TODO: solve padding problems with resnet
        # If the shape of the input map `skip` is not a multiple of 2,
        # it will not match the shape of the upsampled map `upsampled`.
        # If the downsampling uses ceil_mode=False, we nedd to crop `skip`.
        # If it uses ceil_mode=True (not supported here), we should pad it.
        _, _, hu, wu = upsampled.shape
        _, _, hs, ws = skip.shape
        assert (hu <= hs) and (wu <= ws), 'Using ceil_mode=True in pooling?'
        # assert (hu == hs) and (wu == ws), 'Careful about padding'
        if (hs <= hu) and (ws <= wu):
            p2d = (0, wu-ws, 0, hu-hs)
            skip = nn.functional.pad(skip, p2d, "replicate")
        else:
            skip = skip[:, :, :hu, :wu]
        return self.layers(torch.cat([upsampled, skip], dim=1))

class AdaptationBlock(nn.Sequential):
    """ Used to compress the encoder output to fewer dimensions.
    Speed up the optimization by reducing the number of channels.
    """
    def __init__(self, inp, out):
        conv = nn.Conv2d(inp, out, kernel_size=1, padding=0, bias=True)
        super().__init__(conv)

class RetrievalBlock(nn.Module):
    def __init__(self,  lwhiten, pool, whiten):
        super().__init__()
        self.lwhiten = lwhiten
        self.pool = pool
        self.whiten = whiten
        self.norm = L2N()

    def forward(self, x):
        if self.lwhiten is not None:
            s = x.size()
            x = x.permute(0,2,3,1).contiguous().view(-1, s[1])
            x = self.lwhiten(x)
            x = x.view(s[0],s[2],s[3],self.lwhiten.out_features).permute(0,3,1,2)

        # features -> pool -> norm
        x = self.norm(self.pool(x)).squeeze(-1).squeeze(-1)

        # if whiten exist: pooled features -> whiten -> norm
        if self.whiten is not None:
            x = self.norm(self.whiten(x))

        # permute so that it is Dx1 column vector per image (DxN if many images)
        return x.permute(1,0)

class LocalizationHeadBlock(nn.Module):
    def __init__(self, conf):
        super().__init__()

        in_channels = 256
        layers = []
        Block = checkpointed(torch.nn.Sequential, do=conf.checkpointed)

        # Manual definition of 1 block: temporary solution
        blocks = [[]]

        blocks[0].append(nn.MaxPool2d(kernel_size=2, stride=2))
        blocks[0].append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        blocks[0].append(nn.ReLU(inplace=True))
        blocks[0].append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        blocks[0].append(nn.ReLU(inplace=True))
        blocks[0].append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        blocks[0].append(nn.ReLU(inplace=True))
        
        self.head = [Block(*b) for b in blocks]
        self.head = nn.ModuleList(self.head)

    def forward(self, x):

        feat1 = self.head[0](x)

        return feat1

class UNet(BaseModel):
    default_conf = {
        'output_scales': [0, 2, 4],  # what scales to adapt and output
        'output_dim': 128,  # # of channels in output feature maps
        'encoder': 'vgg16',  # string (torchvision net) or list of channels
        'num_downsample': 4,  # how many downsample block (if VGG-style net)
        'decoder': [64, 64, 64, 64],  # list of channels of decoder
        'decoder_norm': 'nn.BatchNorm2d',  # normalization ind decoder blocks
        'do_average_pooling': False,
        'compute_uncertainty': True,
        'checkpointed': False,  # whether to use gradient checkpointing
        'training': True,
        'retrieval': True,
        'load_retrieval': False,
        'retrieval_name': None,
        'normalize': False,
        'two_head': True,
        'rank': False,
        'light': False,
    }
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def build_encoder(self, conf):
        assert isinstance(conf.encoder, str)
        Encoder = getattr(torchvision.models, conf.encoder)
        print('>>> Encoder:', conf.encoder, 'UNet without retrieval loaded')
        encoder = Encoder(pretrained=True)
        if conf.load_retrieval and conf.training:
            print('>>> Retrieval initialization...')
            # gem = Path(__file__).parent / '../../../retrieval/' / conf.retrieval_name
            gem = Path(__file__).parent / '../../retrieval/checkpoints/' / conf.retrieval_name 
            print(gem)
            gem_ckpt = torch.load(str(gem), map_location='cpu')
            
            if conf.encoder.startswith('resnet'):
                torch_dict = encoder.state_dict()
                new_dict = OrderedDict()
                for (k,v), (k2,v2) in zip(torch_dict.items(), gem_ckpt['state_dict'].items()):
                    if k.startswith('fc'):
                        new_dict[k] = v
                    else:
                        new_dict[k] = v2
            else:
                new_dict = OrderedDict((k.replace('encoder', 'features'), v)
                    for k, v in gem_ckpt['state_dict'].items())
        
            encoder.load_state_dict(new_dict, strict = False)
            p = gem_ckpt['state_dict']['pool.p'].item()
        else:
            p = 3.0

        Block = checkpointed(torch.nn.Sequential, do=conf.checkpointed)

        if conf.encoder.startswith('vgg'):
            # Parse the layers and pack them into downsampling blocks
            # It's easy for VGG-style nets because of their linear structure.
            # This does not handle strided convs and residual connections
            assert max(conf.output_scales) <= conf.num_downsample
            skip_dims = []
            previous_dim = None
            blocks = [[]]
            for i, layer in enumerate(encoder.features):
                if isinstance(layer, torch.nn.Conv2d):
                    previous_dim = layer.out_channels
                elif isinstance(layer, torch.nn.MaxPool2d):
                    assert previous_dim is not None
                    skip_dims.append(previous_dim)
                    if (conf.num_downsample + 1) == len(blocks):
                        break
                    blocks.append([])
                    if conf.do_average_pooling:
                        assert layer.dilation == 1
                        layer = torch.nn.AvgPool2d(
                            kernel_size=layer.kernel_size, stride=layer.stride,
                            padding=layer.padding, ceil_mode=layer.ceil_mode,
                            count_include_pad=False)
                blocks[-1].append(layer)
            assert (conf.num_downsample + 1) == len(blocks)
            encoder = [Block(*b) for b in blocks]
        elif conf.encoder.startswith('resnet'):
            # Manually define the splits - this could be improved
            assert conf.encoder[len('resnet'):] in ['18', '34', '50', '101']
            block1 = torch.nn.Sequential(encoder.conv1, encoder.bn1,
                                         encoder.relu)
            block2 = torch.nn.Sequential(encoder.maxpool, encoder.layer1)
            block3 = encoder.layer2
            block4 = encoder.layer3
            blocks = [block1, block2, block3, block4]
            encoder = [torch.nn.Identity()] + [Block(b) for b in blocks]
            skip_dims = [3, 64, 256, 512, 1024]
        else:
            raise NotImplementedError(conf.encoder)

        encoder = nn.ModuleList(encoder)
        return encoder, skip_dims, p

    def _init(self, conf):
        # Encoder
        self.encoder, skip_dims, p = self.build_encoder(conf)

         # Two-heads
        if conf.two_head:
            self.head = LocalizationHeadBlock(conf)

        # Decoder
        if conf.decoder is not None:
            assert len(conf.decoder) == (len(skip_dims) - 1)
            Block = checkpointed(DecoderBlock, do=conf.checkpointed)
            norm = eval(conf.decoder_norm) if conf.decoder_norm else None

            previous = skip_dims[-1]
            decoder = []
            for out, skip in zip(conf.decoder, skip_dims[:-1][::-1]):
                decoder.append(Block(previous, skip, out, norm=norm))
                previous = out
            self.decoder = nn.ModuleList(decoder)

        # Adaptation layers
        adaptation = []
        if conf.compute_uncertainty:
            uncertainty = []
        for idx, i in enumerate(conf.output_scales):
            if conf.decoder is None or i == (len(self.encoder) - 1):
                input_ = skip_dims[i]
            else:
                input_ = conf.decoder[-1-i]

            # out_dim can be an int (same for all scales) or a list (per scale)
            dim = conf.output_dim
            if not isinstance(dim, int):
                dim = dim[idx]

            block = AdaptationBlock(input_, dim)
            adaptation.append(block)
            if conf.compute_uncertainty:
                uncertainty.append(AdaptationBlock(input_, 1))
        self.adaptation = nn.ModuleList(adaptation)
        self.scales = [2**s for s in conf.output_scales]
        if conf.compute_uncertainty:
            self.uncertainty = nn.ModuleList(uncertainty)
        
        if conf.rank: 
            print('Will re-rank with s2dhm')
            
        # Retrieval layers (GeM)
        if conf.retrieval: 
            lwhiten = None
            pooling = GeM(p)
            whiten = None
            self.embedding = RetrievalBlock(lwhiten, pooling, whiten)

    def _forward(self, data):
        image = data['image']
        mean, std = image.new_tensor(self.mean), image.new_tensor(self.std)
        image = (image - mean[:, None, None]) / std[:, None, None]
        
        skip_features = []
        enc_features = []
        features = image
        original_forward = self.conf.training is True or self.conf.light is True or self.conf.rank is False

        if original_forward:
            # When training, testing w/o ranking or using light approach, do original forward
            for block in self.encoder[:4]:
                features = block(features)
                enc_features.append(features)
                    
            # Normalization
            if self.conf.normalize:
                for skip, norm in zip(enc_features, self.normalize):
                    skip_features.append(norm(skip)) 
            else:
                skip_features = enc_features

            if self.conf.two_head:
                feat1= self.head(skip_features[-1])
                skip_features.append(feat1)

                for block in self.encoder[-1]:
                    # Retrieval encoder doesn't take normalized input
                    features = block(features)

                last_feature = features

            else:
                features = block(features)
                skip_features.append(features)
                last_feature = skip_features[-1]

            if self.conf.rank:
                hypercolumn = self.build_hypercolumn_light(skip_features[-3:])
        else:
            # In test, get features and hypercolumn from one forward
            skip_features, hypercolumn = self.build_hypercolumn_one_forward(image, self.encoder, self.head)

            last_feature = skip_features[-2]

            for block in self.encoder[-1]:
                last_feature = block(last_feature)
            
        if self.conf.decoder:
            pre_features = [skip_features[-1]]

            for block, skip in zip(self.decoder, skip_features[:-1][::-1]):
                pre_features.append(block(pre_features[-1], skip))
            pre_features = pre_features[::-1]  # fine to coarse
        else:
            pre_features = skip_features

        out_features = []
        for adapt, i in zip(self.adaptation, self.conf.output_scales):
            out_features.append(adapt(pre_features[i]))
        pred = {'feature_maps': out_features}

        if self.conf.compute_uncertainty:
            confidences = []
            for layer, i in zip(self.uncertainty, self.conf.output_scales):
                unc = layer(pre_features[i])
                conf = torch.sigmoid(-unc)
                confidences.append(conf)
            pred['confidences'] = confidences


        if self.conf.retrieval: 
            embedding = self.embedding(last_feature)
            pred['embedding'] = embedding

        if self.conf.rank: 
            pred['hypercolumn'] = hypercolumn
        else:
            pred['hypercolumn'] = None

        return pred

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError

    def build_hypercolumn(self, image, encoder, head):
        layers_enc = [14, 17, 21]
        layers_head = [1,5]

        feature_maps, j = [], 0
        layer_counter = 0
        feature_map = image

        for block in encoder[:4]:
            for layer in block:
                if len(layers_enc) > j:
                    if(layer_counter==layers_enc[j]):
                        feature_maps.append(feature_map)
                        j+=1
                feature_map = layer(feature_map)
                layer_counter += 1

        head_counter = 0
        h = 0
        for layer in head.head[0]:
            if len(layers_head) > h:
                if(head_counter==layers_head[h]):
                    feature_maps.append(feature_map)
                    h+=1
            feature_map = layer(feature_map)
            head_counter += 1

        # Final descriptor size (concat. intermediate features)
        final_descriptor_size = sum([x.shape[1] for x in feature_maps])
        b, c, w, h = feature_maps[0].shape
        hypercolumn = torch.zeros(b, final_descriptor_size, w, h)

        # Upsample to the largest feature map size
        start_index = 0
        for j in range(len(layers_enc)+len(layers_head)):
            descriptor_size = feature_maps[j].shape[1]
            upsampled_map = nn.functional.interpolate(
                feature_maps[j], size=(w, h),
                mode='bilinear', align_corners=True)
            hypercolumn[:, start_index:start_index + descriptor_size, :, :] = upsampled_map
            start_index += descriptor_size

        # Delete and empty cache
        del feature_maps, feature_map, upsampled_map
        torch.cuda.empty_cache()

        # Normalize descriptors
        hypercolumn = hypercolumn / torch.norm(
            hypercolumn, p=2, dim=1, keepdim=True)
        
        return hypercolumn

    def build_hypercolumn_one_forward(self, image, encoder, head):
        layers_enc = [14, 17, 21]
        layers_head = [1,5]

        feature_maps, j = [], 0
        layer_counter = 0
        feature_map = image
        skip_features = []

        for block in encoder[:4]:
            for layer in block:
                if len(layers_enc) > j:
                    if(layer_counter==layers_enc[j]):
                        feature_maps.append(feature_map)
                        j+=1
                feature_map = layer(feature_map)
                layer_counter += 1
            skip_features.append(feature_map)

        head_counter = 0
        h = 0
        for layer in head.head[0]:
            if len(layers_head) > h:
                if(head_counter==layers_head[h]):
                    feature_maps.append(feature_map)
                    h+=1
            feature_map = layer(feature_map)
            head_counter += 1

        skip_features.append(feature_map)

        # Final descriptor size (concat. intermediate features)
        final_descriptor_size = sum([x.shape[1] for x in feature_maps])
        b, c, w, h = feature_maps[0].shape
        hypercolumn = torch.zeros(b, final_descriptor_size, w, h)

        # Upsample to the largest feature map size
        start_index = 0
        for j in range(len(layers_enc)+len(layers_head)):
            descriptor_size = feature_maps[j].shape[1]
            upsampled_map = nn.functional.interpolate(
                feature_maps[j], size=(w, h),
                mode='bilinear', align_corners=True)
            hypercolumn[:, start_index:start_index + descriptor_size, :, :] = upsampled_map
            start_index += descriptor_size

        # Delete and empty cache
        del feature_maps, feature_map, upsampled_map
        torch.cuda.empty_cache()

        # Normalize descriptors
        hypercolumn = hypercolumn / torch.norm(
            hypercolumn, p=2, dim=1, keepdim=True)
        
        return skip_features, hypercolumn

    def build_hypercolumn_light(self, hyper_maps):

        # Final descriptor size (concat. intermediate features)
        final_descriptor_size = sum([x.shape[1] for x in hyper_maps])
        b, c, w, h = hyper_maps[0].shape
        hypercolumn = torch.zeros(b, final_descriptor_size, w, h)

        # Upsample to the largest feature map size
        start_index = 0
        for j in range(len(hyper_maps)):
            descriptor_size = hyper_maps[j].shape[1]
            upsampled_map = nn.functional.interpolate(
                hyper_maps[j], size=(w, h),
                mode='bilinear', align_corners=True)
            hypercolumn[:, start_index:start_index + descriptor_size, :, :] = upsampled_map
            start_index += descriptor_size

        # Delete and empty cache
        del hyper_maps, upsampled_map
        torch.cuda.empty_cache()

        # Normalize descriptors
        hypercolumn = hypercolumn / torch.norm(
            hypercolumn, p=2, dim=1, keepdim=True)
        
        return hypercolumn

    def build_hypercolumn_decoder(self, encoder_maps, decoder_maps):

        hyper_maps = encoder_maps + decoder_maps

        # Final descriptor size (concat. intermediate features)
        final_descriptor_size = sum([x.shape[1] for x in hyper_maps])
        b, c, w, h = hyper_maps[0].shape
        hypercolumn = torch.zeros(b, final_descriptor_size, w, h)

        # Upsample to the largest feature map size
        start_index = 0
        for j in range(len(hyper_maps)):
            descriptor_size = hyper_maps[j].shape[1]
            upsampled_map = nn.functional.interpolate(
                hyper_maps[j], size=(w, h),
                mode='bilinear', align_corners=True)
            hypercolumn[:, start_index:start_index + descriptor_size, :, :] = upsampled_map
            start_index += descriptor_size

        # Delete and empty cache
        del hyper_maps, upsampled_map
        torch.cuda.empty_cache()

        # Normalize descriptors
        hypercolumn = hypercolumn / torch.norm(
            hypercolumn, p=2, dim=1, keepdim=True)
        
        return hypercolumn
