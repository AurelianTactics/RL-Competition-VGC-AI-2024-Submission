# Version 2 of Submission to VGC AI 2024
* submission_v2 holds the submission
* key files:
    * pequil_bot_battle_policy_v2.py: given a environment state, takes an action
    * simulate_mc_v2_7_14_24.py: runs Monte Carlo first visit like algorithm to gather experience to train the agent
    * create_action_dict_evaluate_v2.ipynb:
        * Based on MC results, judges which actions the agent should take using statistical tests rather than raw reward.
        * Also evaluates final trained agent against other baseline agents
    * join_dict_v2.ipynb: combines MC experience into one dict for analysis
    * pkm_battle_env_wrapper.py: code for baseline agents
