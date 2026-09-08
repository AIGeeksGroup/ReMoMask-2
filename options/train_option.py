from options.base_option import BaseOptions

class TrainT2MOptions(BaseOptions):

    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=64)
        self.parser.add_argument('--max_epoch', type=int, default=500)
        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--gamma', type=float, default=0.1)
        self.parser.add_argument('--milestones', default=[1000000], nargs='+', type=int)
        self.parser.add_argument('--warm_up_iter', default=2000, type=int)
        self.parser.add_argument('--cond_drop_prob', type=float, default=0.1)
        self.parser.add_argument('--seed', default=3407, type=int)
        self.parser.add_argument('--is_continue', action='store_true')
        self.parser.add_argument('--log_every', type=int, default=50)
        self.parser.add_argument('--eval_every_e', type=int, default=10)
        self.parser.add_argument('--train_split', type=str, default='train.txt')
        self.parser.add_argument('--val_split', type=str, default='val.txt')
        self.is_train = True
