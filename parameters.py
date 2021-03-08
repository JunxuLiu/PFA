
class Parameters:
    def __init__(project_path, 
                 N,
                 noniid, 
                 max_steps, 
                 local_steps, 
                 client_dataset_size, 
                 client_batch_size, 
                 num_microbatches,
                 learning_rate_mode=None, 
                 learning_rate=None, 
                 epsilons=None,
                 delta=None,
                 l2_norm_clip=None,
                 sample_mode=None,
                 sample_ratio=None,
                 dpsgd=False,
                 wavg=False,
                 fedavg=False,
                 projection=True,
                 proj_dims=None,
                 lanczos_dims=None,
                 save_dir=None,
                 log_dir=None):

        self.N = N
        self.noniid = noniid
        self.max_steps = max_steps
        
        self.dpsgd = dpsgd
        if self.dpsgd:
            set_dpsgd_parameters(epsilons, delta, l2_norm_clip)

        self.fedavg = fedavg
        self.wavg = wavg
        self.projection = projection
        if self.projection:
            set_projection_parameters(proj_dims, lanczos_iters)
        
        set_client_selection(sample_mode, sample_ratio)

        self.save_dir = os.path.join(project_path, 'res')
        self.log_dir = os.path.join(project_path, 'log')


    def set_client_selection(sample_mode, sample_ratio):
        self.sample_mode = sample_mode
        self.sample_ratio = sample_ratio


    def set_dpsgd_parameters(epsilons, delta, l2_norm_clip):
        self.epsilons = epsilons
        self.delta = delta
        self.l2_norm_clip = l2_norm_clip

        
    def set_loc_train_parameters(local_steps, 
                                 client_dataset_size, 
                                 client_batch_size,
                                 num_microbatches, 
                                 learning_rate_mode,
                                 learning_rate):

        self.local_steps = local_steps
        self.client_dataset_size = client_dataset_size
        self.client_batch_size = client_batch_size
        self.num_microbatches = num_microbatches
        self.learning_rate_mode = learning_rate_mode
        self.learning_rate = learning_rate


    def set_projection_parameters(proj_dims, lanczos_iters):
        self.proj_dims = proj_dims
        self.lanczos_iters = lanczos_iters

