pretrained_model = {'None' : 'none',}


class LoadConfig(object):
    def __init__(self, data, dataset, swap_num, backbone, version):
        if dataset == 'CUB':
            self.numcls = 200
        elif dataset == 'STCAR':
            self.numcls = 196
        else:
            raise Exception('dataset not defined ???')

        self.swap_num = swap_num

        self.backbone = backbone

        self.use_dcl = True
        self.use_backbone = False if self.use_dcl else True
        self.use_Asoftmax = False
        self.use_focal_loss = False
        self.use_fpn = False
        self.use_hier = False

        self.weighted_sample = False
        self.cls_2 = True
        self.cls_2xmul = False




