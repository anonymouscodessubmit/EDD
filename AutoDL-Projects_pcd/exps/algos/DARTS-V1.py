##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
########################################################
# DARTS: Differentiable Architecture Search, ICLR 2019 #
########################################################
import sys, time, random, argparse
from copy import deepcopy
import torch
import math
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets     import get_datasets, get_nas_search_loaders
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from models       import get_cell_based_tiny_net, get_search_spaces
from nas_201_api  import NASBench201API as API
from scipy.stats import entropy

torch.set_printoptions(sci_mode=False)

def search_func(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, logger, gradient_clip,warmup,epoch,softmax_temp,C):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  network.train()
  end = time.time()
  outgrad=0

  def softmaxOutGrad(grad):
    nonlocal outgrad
    outgrad=grad

  #softmax_list=[softmax_temp*10,softmax_temp]
  #softmax_rand=(torch.rand(len(xloader))<C).int()

  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(xloader):
    scheduler.update(None, 1.0 * step / len(xloader))
    base_targets = base_targets.cuda(non_blocking=True)
    arch_targets = arch_targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)
    
    # update the weights
    w_optimizer.zero_grad()
    _, logits= network(base_inputs)
    #_, logits, _= network(base_inputs)
    base_loss = criterion(logits, base_targets)
    base_loss.backward()
    if gradient_clip > 0: torch.nn.utils.clip_grad_norm_(network.parameters(), gradient_clip)
    w_optimizer.step()
    # record
    base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
    base_losses.update(base_loss.item(),  base_inputs.size(0))
    base_top1.update  (base_prec1.item(), base_inputs.size(0))
    base_top5.update  (base_prec5.item(), base_inputs.size(0))
    arch_loss=base_loss

        # update the architecture-weight
    if warmup<=epoch:
        a_optimizer.zero_grad()
        _, logits= network(arch_inputs)

        arch_loss = criterion(logits, arch_targets)
        arch_loss.backward()

        '''
        _, logits, alphas = network(arch_inputs,softmax_temp)
        arch_loss = C*criterion(logits, arch_targets)
        arch_loss.backward()
        '''

        a_optimizer.step()
    # record
    arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
    arch_losses.update(arch_loss.item(),  arch_inputs.size(0))
    arch_top1.update  (arch_prec1.item(), arch_inputs.size(0))
    arch_top5.update  (arch_prec5.item(), arch_inputs.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if step % print_freq == 0 or step + 1 == len(xloader):
      Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=base_losses, top1=base_top1, top5=base_top5)
      Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=arch_losses, top1=arch_top1, top5=arch_top5)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Astr)
  #logger.log('{:}'.format(network.module.show_alphas_grad()))
  #logger.log('softmax-out grad:\n{:}'.format(outgrad))
  return base_losses.avg, base_top1.avg, base_top5.avg


def valid_func(xloader, network, criterion,new_temp):
  data_time, batch_time = AverageMeter(), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  network.eval()
  end = time.time()
  with torch.no_grad():
    for step, (arch_inputs, arch_targets) in enumerate(xloader):
      arch_targets = arch_targets.cuda(non_blocking=True)
      # measure data loading time
      data_time.update(time.time() - end)
      # prediction
      _, logits= network(arch_inputs,new_temp)
      arch_loss = criterion(logits, arch_targets)
      # record
      arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
      arch_losses.update(arch_loss.item(),  arch_inputs.size(0))
      arch_top1.update  (arch_prec1.item(), arch_inputs.size(0))
      arch_top5.update  (arch_prec5.item(), arch_inputs.size(0))
      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()
  return arch_losses.avg, arch_top1.avg, arch_top5.avg


def main(xargs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( xargs.workers )
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
  #config_path = 'configs/nas-benchmark/algos/DARTS.config'
  config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
  search_loader, _, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset, 'configs/nas-benchmark/', config.batch_size, xargs.workers)
  logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader), len(valid_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

  #NAS_BENCH_201         = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
  search_space = get_search_spaces('cell', xargs.search_space_name)
  if xargs.model_config is None:
    model_config = dict2config({'name': 'DARTS-V1', 'C': xargs.channel, 'N': xargs.num_cells,
                                'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                'space'    : search_space,
                                'affine'   : False, 'track_running_stats': bool(xargs.track_running_stats)}, None)
  else:
    model_config = load_config(xargs.model_config, {'num_classes': class_num, 'space'    : search_space,
                                                    'affine'     : False, 'track_running_stats': bool(xargs.track_running_stats)}, None)
  # ../../lib/models/cell_searchs/search_model_darts.py
  search_model = get_cell_based_tiny_net(model_config)
  #logger.log('search-model :\n{:}'.format(search_model))
  
  w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.get_weights(), config)
  a_optimizer = torch.optim.Adam(search_model.get_alphas(), lr=xargs.arch_learning_rate, betas=(0.5, 0.999), weight_decay=xargs.arch_weight_decay)
  logger.log('w-optimizer : {:}'.format(w_optimizer))
  logger.log('a-optimizer : {:}'.format(a_optimizer))
  logger.log('w-scheduler : {:}'.format(w_scheduler))
  logger.log('criterion   : {:}'.format(criterion))
  flop, param  = get_model_infos(search_model, xshape)
  #logger.log('{:}'.format(search_model))
  logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
  if xargs.arch_nas_dataset is None:
    api = None
  else:
    api = API(xargs.arch_nas_dataset)
  logger.log('{:} create API = {:} done'.format(time_string(), api))

  last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
  network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()

  if last_info.exists(): # automatically resume from previous checkpoint
    logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
    last_info   = torch.load(last_info)
    start_epoch = last_info['epoch']
    checkpoint  = torch.load(last_info['last_checkpoint'])
    genotypes   = checkpoint['genotypes']
    valid_accuracies = checkpoint['valid_accuracies']
    search_model.load_state_dict( checkpoint['search_model'] )
    w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
    w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
    a_optimizer.load_state_dict ( checkpoint['a_optimizer'] )
    logger.log("=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
  else:
    logger.log("=> do not find the last-info file : {:}".format(last_info))
    start_epoch, valid_accuracies, genotypes = 0, {'best': -1}, {-1: search_model.genotype()}

  # start training
  start_time, search_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup
  #softmax_para_per_epoch=[]

  softmax_temp=1
  new_temp=softmax_temp
  softmax_decay=0 #每回合要衰减的e^x数量
  #softmax_decay=0.05 #for 4 ops
  decay_rate=0.01
  decay_interval=5
  para_positive_expectation=0.000395 #参数初始化在x正半轴的期望,用来确定初始para的scale
  cycle_interval=10
  decay_count=1
  entropy_n=0
  decay_rate=0.5

  C=0
  warmup=2

  for epoch in range(start_epoch, total_epoch):
    '''
    new_temp=softmax_temp
    if epoch>=warmup: 
        softmax_temp=0.001
        C+=0.8/(50-warmup)
    new_temp=softmax_temp
    if epoch>9 and epoch%decay_interval==0:
        #C=0.5+epoch/100
        with torch.no_grad():
            softmax_out=torch.nn.functional.softmax(search_model.arch_parameters/new_temp, dim=-1).cpu().data.numpy()
            entropy_n=entropy(softmax_out,axis=1).sum()
            if epoch<=39:
                softmax_decay=0.01*entropy_n*decay_rate+(1-decay_rate)*softmax_decay
                new_temp=1/(math.log(softmax_temp+softmax_decay*decay_count)/0.000395)
                decay_count+=1
            else:
                decay_interval=1
                softmax_decay=0.02*entropy_n*decay_rate+(1-decay_rate)*softmax_decay
                new_temp=1/(math.log(softmax_temp+softmax_decay*decay_count)/0.000395)
                decay_count+=1
    '''

    w_scheduler.update(epoch, 0.0)
    need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.val * (total_epoch-epoch), True) )
    epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
    logger.log('\n[Search the {:}-th epoch] {:}, LR={:}, temperature={:}, C={:}'.format(epoch_str, need_time, min(w_scheduler.get_lr()),new_temp,C))

    search_w_loss, search_w_top1, search_w_top5 = search_func(search_loader, network, criterion, w_scheduler, w_optimizer, a_optimizer, epoch_str, xargs.print_freq, logger, xargs.gradient_clip,args.warm_up,epoch,new_temp,C)
    search_time.update(time.time() - start_time)
    logger.log('[{:}] searching : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s'.format(epoch_str, search_w_loss, search_w_top1, search_w_top5, search_time.sum))
    #valid_a_loss , valid_a_top1 , valid_a_top5  = valid_func(valid_loader, network, criterion,new_temp)
    valid_a_loss , valid_a_top1 , valid_a_top5  = valid_func(valid_loader, network, criterion,0.0001)
    logger.log('[{:}] evaluate  : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, valid_a_loss, valid_a_top1, valid_a_top5))
    # check the best accuracy
    valid_accuracies[epoch] = valid_a_top1
    if valid_a_top1 > valid_accuracies['best']:
      valid_accuracies['best'] = valid_a_top1
      genotypes['best']        = search_model.genotype()
      find_best = True
    else: find_best = False

    genotypes[epoch] = search_model.genotype()
    logger.log('<<<--->>> The {:}-th epoch : {:}'.format(epoch_str, genotypes[epoch]))
    # save checkpoint
    save_path = save_checkpoint({'epoch' : epoch + 1,
                'args'  : deepcopy(xargs),
                'search_model': search_model.state_dict(),
                'w_optimizer' : w_optimizer.state_dict(),
                'a_optimizer' : a_optimizer.state_dict(),
                'w_scheduler' : w_scheduler.state_dict(),
                'genotypes'   : genotypes,
                'valid_accuracies' : valid_accuracies},
                model_base_path, logger)
    last_info = save_checkpoint({
          'epoch': epoch + 1,
          'args' : deepcopy(args),
          'last_checkpoint': save_path,
          }, logger.path('info'), logger)
    if find_best:
      logger.log('<<<--->>> The {:}-th epoch : find the highest validation accuracy : {:.2f}%.'.format(epoch_str, valid_a_top1))
      copy_checkpoint(model_base_path, model_best_path, logger)
    with torch.no_grad():
      #logger.log('arch-parameters :\n{:}'.format( nn.functional.softmax(search_model.arch_parameters, dim=-1).cpu() ))
      logger.log('{:}'.format(search_model.show_alphas(new_temp)))
      logger.log('{:}'.format(search_model.show_alphas_grad()))
      #softmax_para_per_epoch.append(search_model.get_softmax())
    if api is not None: logger.log('{:}'.format(api.query_by_arch(genotypes[epoch], '200')))
    # measure elapsed time
    with torch.no_grad():
         softmax_out=torch.nn.functional.softmax(search_model.arch_parameters/new_temp, dim=-1).cpu().data.numpy()
         entropy_n=entropy(softmax_out,axis=1).sum()
    logger.log('entropy={:}'.format(entropy_n))
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  logger.log('\n' + '-'*100)
  logger.log('DARTS-V1 : run {:} epochs, cost {:.1f} s, last-geno is {:}.'.format(total_epoch, search_time.sum, genotypes[total_epoch-1]))
  if api is not None: logger.log('{:}'.format(api.query_by_arch(genotypes[total_epoch-1], '200')))
  logger.close()
  #softmax_para_per_epoch=torch.stack(softmax_para_per_epoch).numpy()
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser("DARTS first order")
  parser.add_argument('--data_path',          type=str,   help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  # channels and number-of-cells
  parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   help='The number of cells in one stage.')
  parser.add_argument('--track_running_stats',type=int,   choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
  parser.add_argument('--config_path',        type=str,   help='The config path.')
  parser.add_argument('--model_config',       type=str,   help='The path of the model configuration. When this arg is set, it will cover max_nodes / channels / num_cells.')
  parser.add_argument('--gradient_clip',      type=float, default=5, help='')
  # architecture leraning rate
  parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
  parser.add_argument('--arch_weight_decay',  type=float, default=1e-3, help='weight decay for arch encoding')
  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--arch_nas_dataset',   type=str,   help='The path to load the architecture dataset (nas-benchmark).')
  parser.add_argument('--print_freq',         type=int,   help='print frequency (default: 200)')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')
  parser.add_argument('--warm_up',          type=int,   default=0, help='manual seed')
  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  main(args)
