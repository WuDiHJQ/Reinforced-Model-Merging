import os
import torch
import argparse
import registry
from copy import deepcopy
from datetime import datetime
from engine.PPO import PPO
from engine.envs import Env_CV, Env_NLP
from engine.datasets.utils import DatasetWrapper
from engine.utils import DataIter, load_clip_features, save_figure
from engine.merge_lib import task_arithmetic, ties, dare, dare_ties


parser = argparse.ArgumentParser(description='RLM Testing')
parser.add_argument('--data_root', default='../data')
parser.add_argument('--model', default='vit_b')
parser.add_argument('--dataset', default='cub,dogs',
                    help='datasets for model merging.')
parser.add_argument('--method', default='ties',
                    help='methods for model merging.')
parser.add_argument('--run_num', default=0, type=int,
                    help='run number for RLM testing.')
parser.add_argument('--workers', default=8, type=int,
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    help='mini-batch size (default: 128)')


def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model.startswith('vit'):
        args.task_type = 'CV'
    else:
        args.task_type = 'NLP'

    datasets = args.dataset.split(',')
    data_iters = []
    base_models = []
    clip_features = []

    for data in datasets:
        _, classes_name,_, val_dst = registry.get_dataset(data, args.data_root)
        if args.task_type == 'CV':
            model = registry.get_model(args.model)
            clip_features.append(load_clip_features(classes_name, device=device))
            val_loader = torch.utils.data.DataLoader(val_dst,
                                                     batch_size=args.batch_size,
                                                     shuffle=True,
                                                     num_workers=args.workers)
        elif args.task_type == 'NLP':
            (tokenizer, model) = registry.get_model(args.model)
            val_dst = DatasetWrapper(val_dst, tokenizer, device)
            val_loader = torch.utils.data.DataLoader(val_dst,
                                                       batch_size=args.batch_size,
                                                       shuffle=False,
                                                       num_workers=args.workers,
                                                       collate_fn=val_dst.collate_fn)

        model = model.to(device).eval()
        model.load_state_dict(torch.load(os.path.join('checkpoint', '%s_%s'%(data, args.model), 'best.pth')))
        data_iters.append(DataIter(val_loader))
        base_models.append(model)

    ################################### Merging ##########################################

    base_ckpt = [model.state_dict() for model in base_models]
    if args.task_type == 'CV':
        pt_ckpt = registry.get_model(args.model).to(device).state_dict()
    elif args.task_type == 'NLP':
        _, pt_model = registry.get_model(args.model)
        pt_ckpt = pt_model.to(device).state_dict()

    if args.method == 'ties':
        merged_state = [ties(base_ckpt,pt_ckpt)]
    elif args.method == 'dare':
        merged_state = [dare(base_ckpt,pt_ckpt)]
    elif args.method == 'dare_ties':
        merged_state = [dare_ties(base_ckpt,pt_ckpt), 
                        ties(base_ckpt,pt_ckpt), 
                        dare(base_ckpt,pt_ckpt)]
    else:
        raise NotImplementedError
    
    merge_models = []
    for state in merged_state:
        model = deepcopy(base_models[0])
        model.load_state_dict(state, strict=False)
        merge_models.append(model)

    ################################### Setting ##########################################
    
    env_name = '%s_%s_%s'%(args.model, args.dataset, args.method)
    run_num = args.run_num

    total_test_episodes = 10        # total num of testing episodes

    K_epochs = 40                   # update policy for K epochs
    eps_clip = 0.2                  # clip parameter for PPO
    gamma = 0.99                    # discount factor

    lr_actor = 0.0003               # learning rate for actor network
    lr_critic = 0.001               # learning rate for critic network

    ###############################################
    
    if args.task_type == 'CV':
        env = Env_CV(base_models, merge_models, data_iters, clip_features, data_scale=1.0)
    elif args.task_type == 'NLP':
        env = Env_NLP(base_models, merge_models, data_iters, data_scale=1.0)
    
    # state/action space dimension
    state_dim = env.observation_space.n
    action_dim = env.action_space.n

    ################################## Testing ###########################################

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device)

    ckpt_path = os.path.join("runs", env_name, 'ckpt_%d.pth'%run_num)
    print("loading network from : " + ckpt_path)
    ppo_agent.load(ckpt_path)

    print("============================================================================================")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        while True:
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")
    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))
    print("============================================================================================")

    ######################################################################################


if __name__ == '__main__':
    main()