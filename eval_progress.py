#!/usr/bin/env python

from network import FeedForwardNN

def _log_summary(ep_len, ep_ret, ep_num):
		# Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration {ep_num} --------------------", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

def rollout(policy, env, render):
	
    obs = env.reset()
    done = False
    t = 0

    # Logging data
    ep_len = 0            # episodic length
    ep_ret = 0            # episodic return

    while not done:
        t += 1
        if render:
            env.render()

        action = policy(obs).detach().numpy()
        obs, rew, done, _ = env.step(action)

        # Sum all episodic rewards as we go along
        ep_ret += rew
        
    ep_len = t

    return ep_len, ep_ret

def eval_progress(list_dir, file_dir, actor_model, env, obs_dim, act_dim, render=False ):
    for entry in list_dir:
        ep_num = entry
        actor_model = file_dir + str(entry) + "/" + actor_model
        policy = FeedForwardNN(obs_dim, act_dim)
        ep_len, ep_ret = rollout(policy, env, render)
        _log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)
    env.close()