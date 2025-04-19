""" Pekiştirmeli Öğrenme (Reinforcement Learning)
Pekiştirmeli öğrenme, bir ajan (model) çevresiyle etkileşime girerek ödül ve ceza alır ve bu geri bildirime dayanarak öğrenir.

Politika Öğrenme (Policy Learning): Ajanın hangi eylemi yapması gerektiğini öğrenmesi.

Değer Öğrenme (Value Learning): Bir eylemin gelecekteki ödülleri nasıl etkilediğini öğrenmesi.

Algoritmalar:
Q-öğrenme (Q-learning)
Derin Q-ağları (DQN)
Politika Gradyan Yöntemleri (Policy Gradient)
Actor-Critic

 1.
 Q - öğrenme(Q - learning) - Basit
 import numpy as np
 import random

 # Aksiyonlar ve durumlar
 actions = [0, 1]  # Örneğin, 0=sola git, 1=sağa git
 states = [0, 1, 2, 3]  # Durumlar

 # Q-matrisinin başlangıç değerleri
 Q = np.zeros((len(states), len(actions)))

 # Parametreler
 learning_rate = 0.1
 discount_factor = 0.9
 epsilon = 0.1
 episodes = 1000


 # Ödül fonksiyonu
 def reward(state, action):
     if state == 3:
         return 100  # Hedefe ulaşmışsa
     else:
         return -1  # Hedefe ulaşamamışsa ceza


 # Q-öğrenme algoritması
 for _ in range(episodes):
     state = random.choice(states)
     while state != 3:
         if random.uniform(0, 1) < epsilon:
             action = random.choice(actions)  # Rastgele aksiyon
         else:
             action = np.argmax(Q[state])  # Maksimum Q değeri olan aksiyon

         next_state = (state + 1) % len(states)  # Durum geçişi
         reward_value = reward(state, action)

         # Q güncelleme
         Q[state, action] = Q[state, action] + learning_rate * (
                     reward_value + discount_factor * np.max(Q[next_state]) - Q[state, action])
         state = next_state

 print("Q-matrix: \n", Q)
 2.
 Derin
 Q - Ağları(Deep
 Q - Network - DQN)
 import gym
 import numpy as np
 import tensorflow as tf
 from tensorflow.keras.models import Sequential
 from tensorflow.keras.layers import Dense
 from collections import deque

 # CartPole environment
 env = gym.make('CartPole-v1')

 # Q-Ağı model
 model = Sequential([
     Dense(24, input_dim=env.observation_space.shape[0], activation='relu'),
     Dense(24, activation='relu'),
     Dense(env.action_space.n, activation='linear')
 ])

 # Hedef model
 target_model = Sequential([
     Dense(24, input_dim=env.observation_space.shape[0], activation='relu'),
     Dense(24, activation='relu'),
     Dense(env.action_space.n, activation='linear')
 ])
 target_model.set_weights(model.get_weights())

 # Hyperparametreler
 gamma = 0.99  # Diskont oranı
 epsilon = 0.1  # Keşif oranı
 learning_rate = 0.001
 episodes = 1000

 # Deneyimler için bellek
 memory = deque(maxlen=2000)

 # Optimizasyon
 optimizer = tf.keras.optimizers.Adam(learning_rate)


 # Q-ağlarını güncelleme fonksiyonu
 def train_model():
     if len(memory) < 32:
         return
     minibatch = random.sample(memory, 32)
     for state, action, reward, next_state, done in minibatch:
         target = reward
         if not done:
             target = reward + gamma * np.max(target_model.predict(next_state[None, :]))
         with tf.GradientTape() as tape:
             q_values = model(state[None, :])
             loss = tf.losses.mean_squared_error(target, q_values[0][action])
         grads = tape.gradient(loss, model.trainable_variables)
         optimizer.apply_gradients(zip(grads, model.trainable_variables))


 # Eğitim
 for e in range(episodes):
     state = env.reset()
     state = np.reshape(state, [1, env.observation_space.shape[0]])
     done = False
     total_reward = 0
     while not done:
         if np.random.rand() <= epsilon:
             action = env.action_space.sample()  # Rastgele aksiyon
         else:
             action = np.argmax(model.predict(state))  # En iyi aksiyon

         next_state, reward, done, _ = env.step(action)
         next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])

         memory.append((state, action, reward, next_state, done))
         state = next_state
         total_reward += reward

         # Modeli güncelle
         train_model()

     # Hedef ağı güncelle
     if e % 10 == 0:
         target_model.set_weights(model.get_weights())

     print(f"Episode {e}/{episodes}, Total Reward: {total_reward}")
 3.
 Politika
 Gradyan
 Yöntemi(Policy
 Gradient)

 import gym
 import numpy as np
 import tensorflow as tf
 from tensorflow.keras.models import Sequential
 from tensorflow.keras.layers import Dense

 # CartPole environment
 env = gym.make('CartPole-v1')

 # Politika Ağı
 model = Sequential([
     Dense(24, input_dim=env.observation_space.shape[0], activation='relu'),
     Dense(24, activation='relu'),
     Dense(env.action_space.n, activation='softmax')
 ])

 # Hyperparametreler
 learning_rate = 0.01
 episodes = 1000
 gamma = 0.99  # Diskont oranı

 # Optimizer
 optimizer = tf.keras.optimizers.Adam(learning_rate)


 # Politika gradyanı güncelleme fonksiyonu
 def update_policy(states, actions, rewards):
     with tf.GradientTape() as tape:
         action_probs = model(np.vstack(states))
         loss = 0
         for i in range(len(states)):
             action_prob = action_probs[i][actions[i]]
             loss -= np.log(action_prob) * rewards[i]  # Negatif log-likelihood
     grads = tape.gradient(loss, model.trainable_variables)
     optimizer.apply_gradients(zip(grads, model.trainable_variables))


 # Eğitim
 for e in range(episodes):
     state = env.reset()
     state = np.reshape(state, [1, env.observation_space.shape[0]])
     done = False
     total_reward = 0
     states, actions, rewards = [], [], []
     while not done:
         # Politika ile aksiyon seç
         action_prob = model(state)
         action = np.random.choice(env.action_space.n, p=action_prob[0])

         next_state, reward, done, _ = env.step(action)
         next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])

         states.append(state)
         actions.append(action)
         rewards.append(reward)

         state = next_state
         total_reward += reward

     # Politika güncelle
     update_policy(states, actions, rewards)
     print(f"Episode {e}/{episodes}, Total Reward: {total_reward}")
 4.
 Actor - Critic Algoritması

 import gym
 import numpy as np
 import tensorflow as tf
 from tensorflow.keras.models import Sequential
 from tensorflow.keras.layers import Dense

 # CartPole environment
 env = gym.make('CartPole-v1')

 # Actor Ağı
 actor_model = Sequential([
     Dense(24, input_dim=env.observation_space.shape[0], activation='relu'),
     Dense(24, activation='relu'),
     Dense(env.action_space.n, activation='softmax')
 ])

 # Critic Ağı
 critic_model = Sequential([
     Dense(24, input_dim=env.observation_space.shape[0], activation='relu'),
     Dense(24, activation='relu'),
     Dense(1)
 ])

 # Hyperparametreler
 learning_rate = 0.01
 episodes = 1000
 gamma = 0.99  # Diskont oranı

 # Optimizer
 actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
 critic_optimizer = tf.keras.optimizers.Adam(learning_rate)


 # Actor-Critic güncelleme fonksiyonu
 def update_actor_critic(states, actions, rewards, next_states, done):
     with tf.GradientTape(persistent=True) as tape:
         action_probs = actor_model(np.vstack(states))
         values = critic_model(np.vstack(states))

         # Geriye doğru ödül
         next_values = critic_model(np.vstack(next_states))
         td_error = rewards + gamma * next_values * (1 - done) - values

         # Actor loss
         action_prob = action_probs[np.arange(len(actions)), actions]
         actor_loss = -tf.reduce_mean(tf.math.log(action_prob) * td_error)

         # Critic loss
         critic_loss = tf.reduce_mean(tf.square(td_error))

     actor_grads = tape.gradient(actor_loss, actor_model.trainable_variables)
     critic_grads = tape.gradient(critic_loss, critic_model.trainable_variables)

     actor_optimizer.apply_gradients(zip(actor_grads, actor_model.trainable_variables))
     critic_optimizer.apply_gradients(zip(critic_grads, critic_model.trainable_variables))


 # Eğitim
 for e in range(episodes):
     state = env.reset()
     state = np.reshape(state, [1, env.observation_space.shape[0]])
     done = False
     total_reward = 0
     states, actions, rewards, next_states, done_flags = [], [], [], [], []
     while not done:
         # Politika ile aksiyon seç
         action_prob = actor_model(state)
         action = np.random.choice(env.action_space.n, p=action_prob[0])

         next_state, reward, done, _ = env.step(action)
         next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])

         states.append(state)
         actions.append(action)
         rewards.append(reward)
         next_states.append(next_state)
         done_flags.append(done)

         state = next_state
         total_reward += reward

     # Actor-Critic güncelle
     update_actor_critic(states, actions, rewards, next_states, done_flags)
     print(f"Episode {e}/{episodes}, Total Reward: {total_reward}")"""