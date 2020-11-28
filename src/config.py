class Config:

  def __init__(self):
    self.task = ''
    # 'kv2seq'
    # 'rnnlm', 
    # 'latent_temp_crf'
    # 'latent_temp_crf_rl'
    # 'autodecoder'
    # 'gaussian_vae' 
    self.model_name = 'latent_temp_crf' 
    self.model_version = 'test'

    self.output_path = '../outputs/'
    self.tensorboard_path = '../tensorboard/'
    self.model_path = '../models/'

    ## Dataset 
    # 'e2e', 'ptb', 'mscoco'
    self.dataset = 'e2e' 
    self.task = 'generation' # 'density' or 'generation' or 
    self.data_path = {
      'e2e': 
        { 'train': '../data/e2e-dataset/trainset.csv', 
          'dev': '../data/e2e-dataset/devset.csv', 
          'test': '../data/e2e-dataset/testset_w_refs.csv',
          'vocab': '../data/e2e-dataset/vocab.txt'},
      'ptb':
        { 'train': '../data/ptb-lm/rnng/train.txt',
          'dev': '../data/ptb-lm/rnng/valid.txt',
          'test': '../data/ptb-lm/rnng/test.txt',
          'vocab': '../data/ptb-lm/rnng/vocab.txt'},
      'mscoco':
        { 'train': '../data/mscoco/captions_train2017.json',
          'test': '../data/mscoco/captions_val2017.json',
          'train_index': '../data/mscoco/mscoco_train_index.txt',
          'vocab': '../data/mscoco/vocab.txt',
          'close_key': '../data/mscoco/close_key.pickle' },
      'sst':
        { 'train': '../data/sst/stsa.binary.train',
          'dev': '../data/sst/stsa.binary.dev',
          'test': '../data/sst/stsa.binary.test'}

      }

    # 95%, do not count _GOO and _EOS
    self.max_sent_len = {'e2e': 37, 'ptb': 39, 'mscoco': 16} 
    # 95% + 1, count _GOO in source and _EOS in target
    self.max_dec_len = {'e2e': 38, 'ptb': 40, 'mscoco': 16} 
    self.max_bow_len = {'e2e': 37, 'ptb': 39, 'mscoco': 16}
    self.max_mem_len = 16 # 99% 

    self.pad_id = 0
    self.start_id = 1
    self.end_id = 2
    self.unk_id = 3
    self.seg_id = 4

    self.word2id = {'_PAD': self.pad_id, '_GOO': self.start_id, 
      '_EOS': self.end_id, '_UNK': self.unk_id, '_SEG': self.seg_id}
    self.id2word = {self.pad_id: '_PAD', self.start_id: '_GOO', 
      self.end_id: '_EOS', self.unk_id: '_UNK', self.seg_id: '_SEG'}

    self.key_vocab_size = -1
    self.vocab_size = -1

    ## Controller 
    # general
    self.is_test = False
    self.test_validate = False
    self.use_tensorboard = True
    self.write_full_predictions = False
    self.device = 'cuda'
    self.gpu_id = '0'
    self.start_epoch = 0
    self.validate_start_epoch = 8
    self.num_epoch = 30
    self.batch_size_train = 500
    self.batch_size_eval = 100
    self.print_interval = 20 
    self.load_ckpt = False
    self.save_ckpt = False # if save checkpoints
    self.all_pretrained_path = ''
    self.lm_pretrained_path = '../models/rnnlm_t.0.0/ckpt_e29'
    self.save_temp = False

    # logging info during training 
    self.log_info = {
      'latent_temp_crf_ar':[
        'loss', 
        'tau', 'x_lambd', 'z_sample_max',
        'reward', 'learning_signal', 
        'p_log_prob', 'p_log_prob_x', 'p_log_prob_z', 'z_acc', 'ppl', 'marginal', 
        'ent_z', 'ent_z_loss', 
        'g_mean', 'g_std', 'g_r'
        ],
      'rnnlm': [
        'loss', 'nll', 'ppl'
        ],
    }

    # models to be inspected during training
    self.inspect_model = True

    # models to inspect gradient 
    self.inspect_grad = False

    # scores to be reported during validation 
    self.validation_scores = {
      'rnnlm': ['ppl', 'nll'],
      'latent_temp_crf_ar': [
        'ent_z', 'elbo', 'marginal', 'p_log_prob', 'z_sample_log_prob', 'ppl', 
        'p_log_prob_x', 'p_log_prob_z', 'z_acc', 
        # 'r1', 'r2', 'rl', 'b1', 'b2', 'b3', 'b4',
        # 'self_r1', 'self_r2', 'self_rl', 'self_b1', 'self_b2', 'self_b3', 'self_b4',
        # 'ir1', 'ir2', 'irl', 'ib1', 'ib2', 'ib3', 'ib4',
        ]
      }

    # validation criteria for different models 
    self.validation_criteria ={
      'latent_temp_crf_ar': 'marginal',
      'rnnlm': 'nll'
      }

    # optimization
    self.seperate_optimizer = False
    self.enc_learning_rate = 1e-4
    self.dec_learning_rate = 1e-4
    self.learning_rate = 1e-4

    # latent z
    self.latent_vocab_size = 50

    self.y_beta = 0. # KLD for y 
    self.bow_beta = 0. # BOW KLD/ entropy regularization
    self.bow_lambd = 1.0 # BOW loss
    self.bow_gamma = 1.0 # stepwise bow loss

    self.z_lambd = 1.0 # learning signal scaling
    self.z_beta = 0.01 # entropy regularization 
    self.z_gamma = 0.01 # switching regularization
    self.z_b0 = 0 # constant baseline 
    self.z_overlap_logits = False # if overlap the z logits
    self.z_lambd_supervised = 1.0 # supervised loss for z 
    self.gumbel_st = True # if use gumbel-straight through estimator 

    # Anneal tau 
    self.z_tau_init = 1.0
    self.z_tau_final = 0.01
    self.tau_anneal_epoch = 40

    self.z_sample_method = 'gumbel_ffbs' # 'gumbel_ffbs', 'pm'
    self.num_sample_nll = 100

    self.stepwise_reward = False
    self.reward_level = 'seq' # 'seq', 'step', 'seq_step', 'no_base'
    self.latent_baseline = 'constant' # 'constant', 'random'
    self.num_sample_rl = 5

    self.grad_estimator = 'reparam' # 'reparam'; 'score_func', 'rnnlm'

    self.dec_adaptive = False 
    self.auto_regressive = True 
    self.use_copy = True  
    self.use_src_info = True

    # anneal word dropout
    self.x_lambd_start_epoch = 10
    self.x_lambd_anneal_epoch = 2

    # decoding 
    self.z_pred_strategy = 'greedy'
    self.x_pred_strategy = 'greedy'
    
    # 'random', 'closest', 'inclusive_closest', 'topk', 'topk-random'
    self.temp_rank_strategy = 'random' 

    # 'greedy', 'sampling_unconstrained', 'sampling_topk', 'sampling_topp_adapt'
    self.decode_strategy = 'greedy'
    self.sampling_topk_k = 2
    self.sampling_topp_gap = 0.1

    self.max_grad_norm = 1.
    self.p_max_grad_norm = 1.
    self.q_max_grad_norm = 5.

    ## model
    # general 
    self.lstm_layers = 1 # 2 for LM on PTB
    self.lstm_bidirectional = True
    self.embedding_size = -1 # the same as state size
    self.state_size = 300 # 650 for LM on PTB
    self.dropout = 0.2 # 0.5 for LM on PTB
    self.copy_decoder = True 
    self.cum_attn = False

    # latent bow
    self.bow_deterministic = False
    self.num_bow_mixture = 3

    # latent template
    self.num_sample = 3
    self.sample_strategy = 'topk' # [topk, unconstrained, greedy]
    self.use_gumbel = False
    self.gumbel_tau = 1.0
    self.stepwise_score = False

  def overwrite(self, args):
    args = vars(args)
    for v in args: setattr(self, v, args[v])
    return 

  def write_arguments(self):
    """Write the arguments to log file"""
    args = vars(self)
    with open(self.output_path + 'arguments.txt', 'w') as fd:
      fd.write('%s_%s\n' % (self.model_name, self.model_version))
      for k in args:
        fd.write('%s: %s\n' % (k, str(args[k])))
    return 

  def print_arguments(self):
    """Print the argument to commandline"""
    args = vars(self)
    print('%s_%s' % (self.model_name, self.model_version))
    for k in args:
      print('%s: %s' % (k, str(args[k])))
    return 
