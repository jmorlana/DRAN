""" Modified from hloc https://github.com/cvg/Hierarchical-Localization """
import sys
from pathlib import Path
import torch
import numpy as np

from ..utils.base_model import BaseModel

dir_path = Path(__file__).parent / '../'
sys.path.append(str(dir_path))

from cirtorch.networks.imageretrievalnet import init_network, extract_ms
from cirtorch.utils.whiten import whitenapply 
from torchvision import transforms

# Implementation of DRAN (Deep Retrieval and deep Alignment Network)
class DRAN(BaseModel):
    default_conf = {
        'dataset': 'sfm120k',
        'checkpoint_dir': dir_path / 'checkpoints',
        'learned_whiten': True,
    }

    def _init(self, conf):
        if conf['dataset'] == 'sfm120k':
            # For Aachen
            checkpoint = conf['checkpoint_dir'] / 'sfm120k.pth.tar'
            checkpoint_whiten = conf['checkpoint_dir'] / 'sfm120k_whiten_ms.pth'
        elif conf['dataset'] == 'msls':
            # For RobotCar-CMU
            checkpoint = conf['checkpoint_dir'] / 'msls.pth.tar'
            checkpoint_whiten = None

        print('>>> Checkpoint is', checkpoint)
        
        # Load checkpoint
        state = torch.load(checkpoint)

        # Parse network parameters
        net_params = {}
        net_params['architecture'] = state['meta']['architecture']
        net_params['pooling'] = state['meta']['pooling']
        net_params['local_whitening'] = state['meta'].get('local_whitening', False)
        net_params['regional'] = state['meta'].get('regional', False)
        net_params['whitening'] = state['meta'].get('whitening', False)
        net_params['mean'] = state['meta']['mean']
        net_params['std'] = state['meta']['std']
        net_params['pretrained'] = False

        # Initialize net
        self.net = init_network(net_params)
        self.net.load_state_dict(state['state_dict'])
        self.net.eval()
        self.net.cuda()  

        # Set whitening
        if conf['learned_whiten']:
            print('Choose to whiten')      
            if checkpoint_whiten:
                print('Load whitening from disk...')
                self.Lw = torch.load(checkpoint_whiten)
            else:
                print('Load whitening from net...')
                self.net.meta['Lw'] = state['meta']['Lw']
                self.Lw = self.net.meta['Lw'][conf['whiten_name']]['ms']

        print(">>>> loaded network: ")
        print(self.net.meta_repr())

        normalize = transforms.Normalize(
            mean=self.net.meta['mean'],
            std=self.net.meta['std']
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    def _forward(self, data):
        image = data['image']
        assert image.shape[1] == 3

        scales = [1, 1/np.sqrt(2), 1/2]
        mean = self.net.meta['mean']
        std = self.net.meta['std']
        image = image - image.new_tensor(mean)[:, None, None]
        image = image / image.new_tensor(std)[:, None, None]

        desc = extract_ms(self.net, image, ms = scales, msp = 1.0).unsqueeze(1)
        desc = desc.squeeze()  # batch dimension

        
        if self.conf['learned_whiten']:
            desc = desc.cpu().numpy()
            desc  = whitenapply(desc.reshape(-1,1), self.Lw['m'], self.Lw['P']).reshape(-1)
            desc = torch.from_numpy(desc)

        
        return {
            'global_descriptor': desc.unsqueeze(0),
        }

    
