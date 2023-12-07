import asyncio
import numpy as np
from tqdm import tqdm

from gym.spaces import Space, Box
from gym.utils.env_checker import check_env
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tabulate import tabulate
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam

from poke_env.environment.battle import AbstractBattle
from poke_env.environment.side_condition import SideCondition
from typechart import TYPECHART
#from poke_env.environment.pokemon_type import TypeChart
#from poke_env.data import TypeChart
#from poke_env.environment import TypeChart
#from poke_env.environment_battle import AbstractBattle
from poke_env.player import (
    background_evaluate_player,
    background_cross_evaluate,
    Gen8EnvSinglePlayer,
    RandomPlayer,
    MaxBasePowerPlayer,
    ObservationType,
    SimpleHeuristicsPlayer,
)


class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        
        reward = self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )
        
        got_status_reward = -1.0
        if current_battle.active_pokemon.status is not None:
            reward += 0.2 * got_status_reward
        
        inflict_status_reward = 1.0
        if last_battle.opponent_active_pokemon.status is None and current_battle.opponent_active_pokemon.status is not None:
            reward += 0.2 * inflict_status_reward
        
        hazard_reward = self.calculate_hazard_reward(last_battle, current_battle)
        reward += 0.3 * hazard_reward
        
        matchup_reward = self.calculate_matchup_reward(current_battle) - self.calculate_matchup_reward(last_battle)
        reward += 0.5 * matchup_reward
        
        return reward
    
    def calculate_matchup_reward(self, battle) -> float:
        reward = 0.0
        
        type_reward = 0.0
        mult1 = battle.active_pokemon.type_1.damage_multiplier(
            battle.opponent_active_pokemon.type_1,
            battle.opponent_active_pokemon.type_2,
            type_chart = TYPECHART
        )
        
        mult2 = 0.0
        if(battle.active_pokemon.type_2 is not None):
            mult2 = battle.active_pokemon.type_2.damage_multiplier(
                battle.opponent_active_pokemon.type_1,
                battle.opponent_active_pokemon.type_2,
                type_chart = TYPECHART
            )
        
        if(mult1 > 1.0):
            type_reward += 1.0
        if(mult2 > 1.0):
            type_reward += 1.0
        reward += type_reward
        
        hp_reward = battle.active_pokemon.current_hp_fraction - battle.opponent_active_pokemon.current_hp_fraction
        reward += hp_reward
        
        return reward
    
    def calculate_hazard_reward(self, last_battle, current_battle) -> float:
        ENTRY_HAZARDS = {
            "spikes": SideCondition.SPIKES,
            "stealhrock": SideCondition.STEALTH_ROCK,
            "stickyweb": SideCondition.STICKY_WEB,
            "toxicspikes": SideCondition.TOXIC_SPIKES,
        }
        
        reward = 0.0
        
        self_last = 0.0
        self_curr = 0.0
        
        opponent_last = 0.0
        opponent_curr = 0.0
        
        for haz in ENTRY_HAZARDS:
            if haz in last_battle.side_conditions:
                self_last += 1.0
            if haz in current_battle.side_conditions:
                self_curr += 1.0
            if haz in last_battle.opponent_side_conditions:
                opponent_last += 1.0
            if haz in last_battle.opponent_side_conditions:
                opponent_curr += 1.0
        
        reward += ((opponent_curr - opponent_last) + (self_last - self_curr))
        
        return reward
        

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart = TYPECHART
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )
        
        own_active = battle.active_pokemon
        opponent_active = battle.opponent_active_pokemon
        
        # Own active Pokemon information
        own_info = [
            own_active.current_hp_fraction,
            own_active.status.value if own_active.status is not None else -1,
        ]
    
        # Team composition information
        team_info = [
            mon.current_hp_fraction for mon in battle.team.values()
        ]
    
        # Opponent's active Pokemon information
        opponent_info = [
            opponent_active.current_hp_fraction,
            opponent_active.status.value if opponent_active.status is not None else -1,
        ]

        # Final vector with 20 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
                own_info,
                team_info,
                opponent_info,
            ]
        )
        
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low_moves = np.full(4, -1.0, dtype=np.float32)
        high_moves = np.full(4, 3.0, dtype=np.float32)
    
        low_dmg_multiplier = np.full(4, -1.0, dtype=np.float32)
        high_dmg_multiplier = np.full(4, 4.0, dtype=np.float32)
    
        low_fainted = np.array([0.0, 0.0], dtype=np.float32)
        high_fainted = np.array([1.0, 1.0], dtype=np.float32)
    
        low_own_info = np.array([-1.0, -1.0], dtype=np.float32)
        high_own_info = np.array([3.0, 1.0], dtype=np.float32)
    
        low_team_info = np.full(6, 0.0, dtype=np.float32)
        high_team_info = np.full(6, 1.0, dtype=np.float32)
    
        low_opponent_info = np.array([-1.0, -1.0], dtype=np.float32)
        high_opponent_info = np.array([3.0, 1.0], dtype=np.float32)
    
        low = np.concatenate([low_moves, low_dmg_multiplier, low_fainted, low_own_info, low_team_info, low_opponent_info])
        high = np.concatenate([high_moves, high_dmg_multiplier, high_fainted, high_own_info, high_team_info, high_opponent_info])
    
        return Box(low, high, dtype=np.float32)


async def main():
    # First test the environment to ensure the class is consistent
    # with the OpenAI API
    #test_env = SimpleRLPlayer(battle_format="gen8randombattle", start_challenging=True)
    #check_env(test_env)
    #test_env.close()

    # Create one environment for training and one for evaluation
    opponent = RandomPlayer(battle_format="gen8randombattle")
    train_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )
    opponent = RandomPlayer(battle_format="gen8randombattle")
    eval_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )

    # Compute dimensions
    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape

    # Create model
    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action, activation="linear"))

    # Defining the DQN
    steps = 10000
    memory = SequentialMemory(limit=steps, window_length=1)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.0,
        nb_steps=100,
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    # Training the model
    dqn.fit(train_env, nb_steps=steps)
    train_env.close()

    # Evaluating the model
    #print("Results against random player:")
    # dqn.test(eval_env, nb_episodes=steps, verbose=True, visualize=False)
    # dqn.test(eval_env, nb_episodes=10000, verbose=False, visualize=False)
    #print(
    #    f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    #)
    #second_opponent = MaxBasePowerPlayer(battle_format="gen8randombattle")
    #eval_env.reset_env(restart=True, opponent=second_opponent)
    #print("Results against max base power player:")
    #dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    #print(
    #    f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    #)
    #eval_env.reset_env(restart=False)

    # Evaluate the player with included util method
    #n_challenges = 250
    #placement_battles = 40
    #eval_task = background_evaluate_player(
    #    eval_env.agent, n_challenges, placement_battles
    #)
    #dqn.test(eval_env, nb_episodes=n_challenges, verbose=False, visualize=False)
    #print("Evaluation with included method:", eval_task.result())
    eval_env.reset_env(restart=False)

    # Cross evaluate the player with included util method
    n_challenges = 50
    players = [
        eval_env.agent,
        RandomPlayer(battle_format="gen8randombattle"),
        MaxBasePowerPlayer(battle_format="gen8randombattle"),
        SimpleHeuristicsPlayer(battle_format="gen8randombattle"),
    ]
    cross_eval_task = background_cross_evaluate(players, n_challenges)
    dqn.test(
        eval_env,
        nb_episodes=n_challenges * (len(players) - 1),
        verbose=True,
        visualize=False,
    )
    cross_evaluation = cross_eval_task.result()
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    print("Cross evaluation of DQN with baselines:")
    print(tabulate(table))
    eval_env.close()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())