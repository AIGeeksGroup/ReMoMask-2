from options.base_option import BaseOptions

class EvalT2MOptions(BaseOptions):

    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--which_epoch', type=str, default='net_best_fid_ep0309.tar')
        self.parser.add_argument('--ext', type=str, default='text2motion')
        self.parser.add_argument('--repeat_times', default=1, type=int)
        self.parser.add_argument('--cond_scale', default=4, type=float)
        self.parser.add_argument('--temperature', default=1.0, type=float)
        self.parser.add_argument('--topkr', default=0.9, type=float)
        self.parser.add_argument('--time_steps', default=18, type=int)
        self.parser.add_argument('--seed', default=10107, type=int)
        self.parser.add_argument('--mtrans_name', type=str, default='v2_mtrans_vgate')
        self.parser.add_argument('--rt_in_value', action='store_true')
        self.parser.add_argument('--ze_database_path', type=str, default='database_ze')
        self.parser.add_argument('--projector_path', type=str, default='logs/query_projector_repair_ep/best_projector.pt')
        self.parser.add_argument('--retrieval_dim', type=int, default=None)
        self.parser.add_argument('--retrieval_topk', type=int, default=1)
        self.parser.add_argument('--retrieval_pool', type=int, default=10)
        self.parser.add_argument('--retr_vgate', action='store_true')
        self.parser.add_argument('--retr_cfgmix_w', type=float, default=None)
        self.parser.add_argument('--clip_version', default='ViT-B/32')
        self.is_train = False
