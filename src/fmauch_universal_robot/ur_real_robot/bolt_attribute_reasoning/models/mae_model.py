import torch
import torch.nn as nn
from util.pos_embed import interpolate_pos_embed
from models import models_vit
from timm.models.layers import trunc_normal_
import util.misc as misc

class OnClassify_v3(nn.Module):

    def __init__(self, args):
        super(OnClassify_v3, self).__init__()

        self.net = models_vit.__dict__[args.model](
            num_classes=2,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )

        if args.finetune and not args.eval:
            checkpoint = torch.load(args.finetune, map_location='cpu')

            print("Load pre-trained checkpoint from: %s" % args.finetune)
            checkpoint_model = checkpoint['model']
            state_dict = self.net.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print("mae_model_27")
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(self.net, checkpoint_model)

            # load pre-trained model
            msg = self.net.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            if args.global_pool:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            else:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

            # manually initialize fc layer
            trunc_normal_(self.net.head.weight, std=2e-5)

        for param in self.net.parameters():
            param.requires_grad = False

        for param in [self.net.head.weight, self.net.head.bias, self.net.fc_norm.weight, self.net.fc_norm.bias]:
            param.requires_grad = True

        self.net.to(args.device)


        model_without_ddp = self.net
        n_parameters = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        for k, v in model_without_ddp.named_parameters():
            print(k)
            print(v.requires_grad)



        print("Model = %s" % str(model_without_ddp))
        print('number of params (M): %.2f' % (n_parameters / 1.e6))

        # if args.distributed:
        #     self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[args.gpu])

    def forward(self, x):
        y_pred = self.net(x)

        return y_pred
