from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from scipy.special import softmax
from v2i.src.core.common import readYaml

class ppoController:

    def __init__(self, sim_config, algoConfig, checkPointPath):

        import ray
        from ray.tune import run_experiments
        from ray.tune.registry import register_env
        from ray.rllib.agents.registry import get_agent_class

        from v2i import V2I

        # Do Essentials
        algoConfig["EXP_NAME"]["config"]["num_workers"] = 2
        algoConfig["EXP_NAME"]["config"]["num_envs_per_worker"] = 1
        algoConfig["EXP_NAME"]["config"]["train_batch_size"] = algoConfig["EXP_NAME"]["config"]["num_workers"] * algoConfig["EXP_NAME"]["config"]["sgd_minibatch_size"]
        simConfigYaml = readYaml(sim_config)
        self.lstmEnabled = False

        if simConfigYaml['config']['enable-lstm']:
            algoConfig['EXP_NAME']['config']['model']['use_lstm'] = True
            self.lstmEnabled = True
        else:
            algoConfig['EXP_NAME']['config']['model']['use_lstm'] = False

        env_creator_name = "v2i-v0"
        register_env(env_creator_name, lambda config: V2I.V2I(sim_config, "train"))

        ray.init()
        cls = get_agent_class('PPO')
        self.agent = cls(env=env_creator_name, config=algoConfig["EXP_NAME"]["config"])
        self.agent.restore(checkPointPath)
        print("Loaded Checkpoint -> %s"%(checkPointPath))
    

    def getAction(self, state, lstm_state):
        if self.lstmEnabled:
            out = self.agent.get_policy(DEFAULT_POLICY_ID).compute_single_action(state, lstm_state)
        else:
            #print("No")
            out = self.agent.get_policy(DEFAULT_POLICY_ID).compute_single_action(state, [])
        
        actionProbs = softmax(out[2]['behaviour_logits'])
        action = out[0]
        lstm_state = out[1]
        vf_preds = out[2]["vf_preds"]
        #action, lstm_state, vf = self.agent.compute_action(state, lstm_state)
        return action, lstm_state, actionProbs, vf_preds