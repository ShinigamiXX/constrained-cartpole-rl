from skopt import gp_minimize
from gym_wrapper import GymWrapper
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import gymnasium as gym
import config
import numpy as np

def tune_hyperparameters_qlearning(model_class, exploration_strategy_class, config):
    # Define the hyperparameter search space
    space = [
        Real(1e-5, 1e-1, name='learning_rate', prior='log-uniform'),
        Real(0.8, 0.999, name='discount_factor'),
        Real(0.01, 0.2, name='epsilon')
    ]

    @use_named_args(space)
    def objective(learning_rate, discount_factor, epsilon):
        # Initialize environment
        env = GymWrapper(gym.make('CartPole-v1'))
        state_dim = env.env.observation_space.shape[0]
        action_dim = env.env.action_space.n
        exploration_strategy = exploration_strategy_class(epsilon=epsilon)
        '''control_params = {
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'epsilon': epsilon
        }'''

        config.CONTROL_PARAMS.update({
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'epsilon': epsilon
        })

        #controller = model_class(control_params, exploration_strategy)

        controller = model_class(config.CONTROL_PARAMS, exploration_strategy, state_dim, action_dim)

        '''controller = model_class(
        control_params={'learning_rate': learning_rate, 'discount_factor': discount_factor},
        exploration_strategy=exploration_strategy
        )'''
        
        total_reward = 0
        # Training loop (simplified for tuning purposes)
        for episode in range(config.NUM_EPISODES):
            state = env.reset()
            episode_reward = 0
            for t in range(config.MAX_STEPS):
                action = controller.get_action(state)
                next_state, reward, done, _ = env.step(action)
                controller.update(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                if done:
                    break
            total_reward += episode_reward

        # Return the negative total reward (since we are minimizing)
        return -total_reward / config.NUM_EPISODES  # Minimize the negative average reward

    # Run Bayesian optimization
    result = gp_minimize(objective, space, n_calls=50, random_state=42)

    # Update config with the best parameters
    config.CONTROL_PARAMS.update({
        'learning_rate': result.x[0],
        'discount_factor': result.x[1],
        'epsilon': result.x[2]
    })

    print("Best Parameters Updated in CONTROL_PARAMS:")
    print(f"Learning Rate: {config.CONTROL_PARAMS['learning_rate']}")
    print(f"Discount Factor: {config.CONTROL_PARAMS['discount_factor']}")
    print(f"Epsilon: {config.CONTROL_PARAMS['epsilon']}")
    print(f"Best Score: {-result.fun}")

def tune_hyperparameters_dqn(model_class, exploration_strategy_class, config):
    # Define the hyperparameter search space
    space = [
    Real(1e-3, 1e-2, name='learning_rate', prior='log-uniform'),
    Real(0.9, 0.999, name='discount_factor'),
    Integer(64, 128, name='batch_size'),
    Real(0.05, 0.2, name='epsilon'),
    Integer(10000, 30000, name='buffer_size'),
    ]

    @use_named_args(space)
    def objective(learning_rate, discount_factor, epsilon, batch_size, buffer_size):
        # Initialize environment
        env = GymWrapper(gym.make('CartPole-v1'))
        state_dim = env.env.observation_space.shape[0]
        action_dim = env.env.action_space.n
        exploration_strategy = exploration_strategy_class(epsilon=epsilon)
        '''control_params = {
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'epsilon': epsilon
        }'''

        config.CONTROL_PARAMS.update({
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'batch_size': batch_size,
            'epsilon': epsilon,
            'buffer_size': buffer_size
        })

        #controller = model_class(control_params, exploration_strategy)

        controller = model_class(config.CONTROL_PARAMS, exploration_strategy, state_dim, action_dim)

        '''controller = model_class(
        control_params={'learning_rate': learning_rate, 'discount_factor': discount_factor},
        exploration_strategy=exploration_strategy
        )'''
        
        total_reward = 0
        # Training loop (simplified for tuning purposes)
        for episode in range(config.NUM_EPISODES):
            state = env.reset()
            episode_reward = 0
            for t in range(config.MAX_STEPS):
                action = controller.get_action(state)
                next_state, reward, done, _ = env.step(action)
                controller.update(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                if done:
                    break
            total_reward += episode_reward

        # Return the negative total reward (since we are minimizing)
        return -total_reward / config.NUM_EPISODES  # Minimize the negative average reward

    # Run Bayesian optimization
    result = gp_minimize(objective, space, n_calls=20, random_state=42)

    print("Optimization result:", result.x)

    # Update config with the best parameters
    config.CONTROL_PARAMS.update({
        'learning_rate': result.x[0],
        'discount_factor': result.x[1],
        'batch_size': result.x[2],
        'epsilon': result.x[3],
        'buffer_size': result.x[4]
    })

    print("Best Parameters Updated in CONTROL_PARAMS:")
    print(f"Learning Rate: {config.CONTROL_PARAMS['learning_rate']}")
    print(f"Discount Factor: {config.CONTROL_PARAMS['discount_factor']}")
    print(f"Epsilon: {config.CONTROL_PARAMS['epsilon']}")
    print(f"Batch Size: {config.CONTROL_PARAMS['batch_size']}")
    print(f"Buffer Size: {config.CONTROL_PARAMS['buffer_size']}")
    print(f"Best Score: {-result.fun}")


def tune_hyperparameters_sarsa(model_class, exploration_strategy_class, config):
    # Define the hyperparameter search space
    space = [
        Real(1e-3, 1e-1, name='learning_rate', prior='log-uniform'),
        Real(0.9, 0.999, name='discount_factor'),
        Real(0.01, 0.2, name='epsilon'),
    ]

    @use_named_args(space)
    def objective(learning_rate, discount_factor, epsilon):
        # Initialize environment
        env = GymWrapper(gym.make('CartPole-v1'))
        exploration_strategy = exploration_strategy_class(epsilon=epsilon)
        
        # Update config with the current hyperparameters
        config.CONTROL_PARAMS.update({
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'epsilon': epsilon
        })

        controller = model_class(config.CONTROL_PARAMS, exploration_strategy)

        total_reward = 0
        # Training loop (simplified for tuning purposes)
        for episode in range(config.NUM_EPISODES):
            state = env.reset()
            action = controller.get_action(state)
            episode_reward = 0
            for t in range(config.MAX_STEPS):
                next_state, reward, done, _ = env.step(action)
                next_action = controller.get_action(next_state)
                controller.update(state, action, reward, next_state, next_action, done)
                state = next_state
                action = next_action
                episode_reward += reward
                if done:
                    break
            total_reward += episode_reward

        # Return the negative total reward (since we are minimizing)
        return -total_reward / config.NUM_EPISODES  # Minimize the negative average reward

    # Run Bayesian optimization
    result = gp_minimize(objective, space, n_calls=50, random_state=42)

    # Update config with the best parameters
    config.CONTROL_PARAMS.update({
        'learning_rate': result.x[0],
        'discount_factor': result.x[1],
        'epsilon': result.x[2],
    })

    print("Best Parameters Updated in CONTROL_PARAMS:")
    print(f"Learning Rate: {config.CONTROL_PARAMS['learning_rate']}")
    print(f"Discount Factor: {config.CONTROL_PARAMS['discount_factor']}")
    print(f"Epsilon: {config.CONTROL_PARAMS['epsilon']}")
    print(f"Best Score: {-result.fun}")

