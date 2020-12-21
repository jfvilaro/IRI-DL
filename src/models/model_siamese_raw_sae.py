import torch
from collections import OrderedDict
from src.utils import util
from .models import BaseModel
from src.networks.networks import NetworksFactory
from src.utils.plots import plot_estim, plot_estim_siamese, concatenate_pair_data, concatenate_trio_data
import numpy as np

class ModelSiameseRawSAE(BaseModel):
    def __init__(self, opt):
        super(ModelSiameseRawSAE, self).__init__(opt)
        self._name = 'ModelSiameseRawSAE'

        # init input params
        self._init_set_input_params()

        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # load networks and optimizers
        if not self._is_train or self._opt["model"]["load_epoch"] > 0:
            self.load()

        # init losses
        if self._is_train:
            self._init_losses()

        # prefetch inputs
        self._init_prefetch_inputs()

    def _init_set_input_params(self):
        self._B = self._opt[self._dataset_type]["batch_size"]               # batch
        self._S = self._opt[self._dataset_type]["image_size"]               # image size
        self._Ci = self._opt[self._dataset_type]["img_nc"]                  # num channels image
        self._Ct = self._opt[self._dataset_type]["target_nc"] * self._B     # num channels target

    def _init_create_networks(self):
        # create reg
        reg_type = self._opt["networks"]["reg"]["type"]
        reg_hyper_params = self._opt["networks"]["reg"]["hyper_params"]
        self._reg = NetworksFactory.get_by_name(reg_type, **reg_hyper_params)
        self._reg = torch.nn.DataParallel(self._reg, device_ids=self._reg_gpus_ids)

    def _init_train_vars(self):
        self._current_lr = self._opt["train"]["reg_lr"]
        self._optimizer = torch.optim.SGD(self._reg.parameters(), lr=self._current_lr)

    def _init_losses(self):
        #self._criterion = torch.nn.MSELoss(reduction='sum').to(self._device_master)

        # total_loss = torch.max(torch.Tensor([0, C-self._dist_diff]))
        self._criterion = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')


    def _init_prefetch_inputs(self):
        self._input_img1 = torch.zeros([self._B, self._Ci, self._S, self._S]).to(self._device_master)
        self._input_img2 = torch.zeros([self._B, self._Ci, self._S, self._S]).to(self._device_master)
        self._input_img3 = torch.zeros([self._B, self._Ci, self._S, self._S]).to(self._device_master)
        self._input_target1 = torch.zeros([self._Ct], dtype=torch.long).to(self._device_master)
        self._input_target2 = torch.zeros([self._Ct], dtype=torch.long).to(self._device_master)

    def set_input(self, input):
        # copy values
        self._input_img1.copy_(input['img1'].view(self._B, self._Ci, self._S, self._S))
        self._input_img2.copy_(input['img2'].view(self._B, self._Ci, self._S, self._S))
        self._input_img3.copy_(input['img3'].view(self._B, self._Ci, self._S, self._S))
        self._input_target1.copy_(input['target1'])
        self._input_target2.copy_(input['target2'])

        # move to gpu
        self._input_img1 = self._input_img1.to(self._device_master)
        self._input_img2 = self._input_img2.to(self._device_master)
        self._input_img3 = self._input_img3.to(self._device_master)
        self._input_target1 = self._input_target1.to(self._device_master)
        self._input_target2 = self._input_target2.to(self._device_master)

    def set_train(self):
        self._reg.train()
        self._is_train = True

    def set_eval(self):
        self._reg.eval()
        self._is_train = False

    def evaluate(self):
        # set model to eval
        is_train = self._is_train
        if is_train:
            self.set_eval()

        # estimate object categories
        with torch.no_grad():
            self.forward(keep_data_for_visuals=True, estimate_loss=False)
            eval = np.transpose(self._vis_input_img, (1, 2, 0))

        # set model back to train if necessary
        if is_train:
            self.set_train()

        return eval

    def evaluate_descriptor(self):
        # set model to eval
        is_train = self._is_train
        if is_train:
            self.set_eval()

        # estimate object categories
        with torch.no_grad():
            ft_ds = self.forward_descriptor()

        # set model back to train if necessary
        if is_train:
            self.set_train()

        return ft_ds

    def optimize_parameters(self, curr_epoch, keep_data_for_visuals=False):
        if self._is_train:

            # calculate loss
            loss1 = self.forward_eq()
            loss2 = self.forward_df()
            loss = loss1 + loss2

            # optimize
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            self._loss_gt = loss

            # keep visuals
            if keep_data_for_visuals:
                self._keep_data(self._dist_equal, self._dist_diff)

        else:
            raise ValueError('Trying to optimize in non-training mode!')

    def forward(self, keep_data_for_visuals=False, estimate_loss=True):

        descriptor_1 = self._estimate(self._input_img1)
        descriptor_2 = self._estimate(self._input_img2)
        descriptor_3 = self._estimate(self._input_img3)

        distance_12 = self._criterion(descriptor_1, descriptor_2)
        distance_23 = self._criterion(descriptor_3, descriptor_2)
        #
        # descriptor_1_squared = torch.pow(descriptor_1, 2)
        # descriptor_2_squared = torch.pow(descriptor_2, 2)
        # descriptor_3_squared = torch.pow(descriptor_3, 2)
        #
        # distance_12 = self._criterion(descriptor_1_squared, descriptor_2_squared)
        # distance_23 = self._criterion(descriptor_2_squared, descriptor_3_squared)

        C = self._opt['model']['constant_c']

        # estimate loss
        if estimate_loss:
            self._dist_equal = distance_12
            self._dist_diff = distance_23
            self._loss_gt = distance_12 + max(0, C-distance_23)  # second term should be max(0, C-dist2)
            total_loss = self._loss_gt
        else:
            total_loss = -1

        # keep visuals
        if keep_data_for_visuals:
            self._keep_data(distance_12, distance_23)

        return total_loss

    def forward_eq(self, estimate_loss=True):

        descriptor_1 = self._estimate(self._input_img1)
        descriptor_2 = self._estimate(self._input_img2)

        #dist_tensor = torch.norm(descriptor_1-descriptor_2, dim=1)
        #mean_dist = torch.mean(dist_tensor, 0)

        distance_12 = self._criterion(descriptor_1, descriptor_2)

        # estimate loss
        if estimate_loss:
            self._dist_equal = distance_12
            total_loss = self._dist_equal
        else:
            total_loss = -1

        return total_loss

    def forward_df(self, estimate_loss=True):

        descriptor_2 = self._estimate(self._input_img2)
        descriptor_3 = self._estimate(self._input_img3)

        #dist_tensor = torch.norm(descriptor_2-descriptor_3, dim=1)
        #mean_dist = torch.mean(dist_tensor, 0)

        C = self._opt['model']['constant_c']

        distance_23 = self._criterion(descriptor_3, descriptor_2)

        # estimate loss
        if estimate_loss:
            self._dist_diff = distance_23
            max_loss = torch.cat([torch.Tensor([0]).to(self._device_master), torch.Tensor([C]).to(self._device_master)-torch.unsqueeze(distance_23,0)])
            total_loss = torch.max(max_loss)  # Second term should be max(0, C-dist2)
        else:
            total_loss = -1

        return total_loss

    def forward_descriptor(self):

        feat_des_1 = self._estimate(self._input_img1)
        feat_des_2 = self._estimate(self._input_img2)
        feat_des_3 = self._estimate(self._input_img3)

        return feat_des_1, feat_des_2, feat_des_3

    def _estimate(self, img):
        return self._reg.forward(img)

    def _keep_data(self, distance_1, distance_2):

        # Concatenate images for visualization

        imga = self._input_img1
        imgb = self._input_img2
        imgc = self._input_img3

        add_border_2_impair = 'True'

        # Plot images trio

        new_img = concatenate_trio_data(imga, imgb, imgc, add_border_2_impair)

        vis_img = util.tensor2im(new_img.detach(), unnormalize=True, to_numpy=True)

        self._vis_input_img = plot_estim_siamese(vis_img, distance_1, distance_2, self._input_target1.detach().cpu().numpy(), self._input_target2.detach().cpu().numpy())

        # Plot images pairs

        # new_img1 = concatenate_pair_data(imga, imgb, add_border_2_impair)
        # new_img2 = concatenate_pair_data(imgb, imgc, add_border_2_impair)


        #vis_img1 = util.tensor2im(new_img1.detach(), unnormalize=True, to_numpy=True)
        #vis_img2 = util.tensor2im(new_img2.detach(), unnormalize=True, to_numpy=True)

        #self._vis_input_img1 = plot_estim_siamese(vis_img1, vis_img1, predicted1, self._input_target1.detach().cpu().numpy())
        #self._vis_input_img2 = plot_estim_siamese(vis_img2, vis_img2, predicted2, self._input_target2.detach().cpu().numpy())

    def get_image_paths(self):
        return OrderedDict()

    def get_current_errors(self):
        loss_dict = OrderedDict()
        loss_dict["loss_gt"] = self._loss_gt.item()
        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([('lr', self._current_lr)])

    def get_current_visuals(self):
        visuals = OrderedDict()
        visuals["1_estim_img"] = self._vis_input_img
        return visuals

    def get_current_dist_equal_gt(self):
        loss_dict = OrderedDict()
        loss_dict["dist_equal_gt"] = self._dist_equal
        return loss_dict

    def get_current_dist_diff_gt(self):
        loss_dict = OrderedDict()
        loss_dict["dist_diff_gt"] = self._dist_diff
        return loss_dict

    def get_current_accuracy(self, accuracy):
        loss_dict = OrderedDict()
        loss_dict["accuracy_gt"] = accuracy
        return loss_dict

    # def get_current_precision_neg(self, precision):
    #     loss_dict = OrderedDict()
    #     loss_dict["precision_neg_gt"] = precision
    #     return loss_dict
    #
    # def get_current_precision_pos(self, precision):
    #     loss_dict = OrderedDict()
    #     loss_dict["precision_pos_gt"] = precision
    #     return loss_dict
    #
    # def get_current_recall_neg(self, recall):
    #     loss_dict = OrderedDict()
    #     loss_dict["recall_neg_gt"] = recall
    #     return loss_dict
    #
    # def get_current_recall_pos(self, recall):
    #     loss_dict = OrderedDict()
    #     loss_dict["recall_pos_gt"] = recall
    #     return loss_dict

    def save(self, epoch_label, save_type, do_remove_prev=True):
        # save networks
        self._save_network(self._reg, 'nn_reg', epoch_label, save_type, do_remove_prev)
        self._save_optimizer(self._optimizer, 'o_reg', epoch_label, save_type, do_remove_prev)

    def load(self):
        # load networks
        load_epoch = self._opt["model"]["load_epoch"]
        self._load_network(self._reg, 'nn_reg', load_epoch)
        if self._is_train:
            self._load_optimizer(self._optimizer, "o_reg", load_epoch)

    def update_learning_rate(self, curr_epoch):
        initial_lr = float(self._opt["train"]["reg_lr"])
        nepochs_no_decay = self._opt["train"]["nepochs_no_decay"]
        nepochs_decay = self._opt["train"]["nepochs_decay"]

        # update lr
        if curr_epoch <= nepochs_no_decay:
            self._current_lr = initial_lr
        else:
            new_lr = self._lr_linear(self._current_lr, nepochs_decay, initial_lr)
            self._update_learning_rate(self._optimizer, "reg", self._current_lr, new_lr)
            self._current_lr = new_lr


