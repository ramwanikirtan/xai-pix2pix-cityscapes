import torch
from torch.cuda.amp import autocast, GradScaler
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    """This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="aligned")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument("--lambda_L1", type=float, default=100.0, help="weight for L1 loss")
            parser.add_argument("--lambda_perceptual", type=float, default=10.0, help="weight for VGG perceptual loss (0 to disable)")

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.mc_dropout = getattr(opt, 'mc_dropout', False)
        # specify the training losses you want to print out
        self.loss_names = ["G_GAN", "G_L1", "G_perceptual", "D_real", "D_fake"]
        if self.isTrain and self.mc_dropout:
            self.loss_names.append("G_diversity")
        # specify the images you want to save/display
        self.visual_names = ["real_A", "fake_B", "real_B"]
        if self.isTrain and self.mc_dropout:
            self.visual_names.append("fake_B2")  # second MC sample for visualization
        # specify the models you want to save to the disk
        if self.isTrain:
            self.model_names = ["G", "D"]
        else:
            self.model_names = ["G"]
        self.device = opt.device
        # MC-dropout requires dropout ON — force no_dropout=False
        use_dropout = not opt.no_dropout if not self.mc_dropout else True
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, use_dropout, opt.init_type, opt.init_gain)

        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # VGG perceptual loss
            self.lambda_perceptual = opt.lambda_perceptual
            if self.lambda_perceptual > 0:
                self.criterionVGG = networks.VGGPerceptualLoss().to(self.device).eval()
            else:
                self.criterionVGG = None
            # MC-dropout diversity loss (lightweight L1-based, no extra network)
            if self.mc_dropout:
                self.lambda_diversity = opt.lambda_diversity
                self.criterionDiversity = networks.DiversityLoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # AMP (mixed precision) setup
            self.use_amp = getattr(opt, 'amp', False)
            self.scaler_G = GradScaler(enabled=self.use_amp)
            self.scaler_D = GradScaler(enabled=self.use_amp)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def _enable_dropout(self):
        """Keep dropout layers active (train mode) for MC-dropout stochastic inference."""
        for m in self.netG.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        with autocast(enabled=getattr(self, 'use_amp', False)):
            self.fake_B = self.netG(self.real_A)  # G(A)
            # Second forward pass with different dropout mask for diversity training
            # Detached — gradients only flow through fake_B, saving VRAM
            if self.isTrain and self.mc_dropout:
                with torch.no_grad():
                    self.fake_B2 = self.netG(self.real_A).detach()

    def forward_mc(self, n_samples):
        """Run N MC-dropout forward passes for diverse test-time outputs."""
        self._enable_dropout()
        samples = []
        with torch.no_grad():
            with autocast(enabled=getattr(self, 'use_amp', False)):
                for _ in range(n_samples):
                    samples.append(self.netG(self.real_A))
        return samples

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        with autocast(enabled=self.use_amp):
            # Fake; stop backprop to the generator by detaching fake_B
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB.detach())
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            # Real
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)
            self.loss_D_real = self.criterionGAN(pred_real, True)
            # combine loss and calculate gradients
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.scaler_D.scale(self.loss_D).backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        with autocast(enabled=self.use_amp):
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            # Second, G(A) = B
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            # Third, perceptual loss
            if self.criterionVGG is not None:
                self.loss_G_perceptual = self.criterionVGG(self.fake_B, self.real_B) * self.lambda_perceptual
            else:
                self.loss_G_perceptual = 0.0
            # Fourth, MC-dropout diversity loss (negative LPIPS → maximize diversity)
            # fake_B2 is detached so grads flow through fake_B only, saving VRAM
            if self.mc_dropout:
                self.loss_G_diversity = self.criterionDiversity(self.fake_B, self.fake_B2) * self.lambda_diversity
            else:
                self.loss_G_diversity = 0.0
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_perceptual + self.loss_G_diversity
        self.scaler_G.scale(self.loss_G).backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.scaler_D.step(self.optimizer_D)  # update D's weights
        self.scaler_D.update()
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.scaler_G.step(self.optimizer_G)  # update G's weights
        self.scaler_G.update()


