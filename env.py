from collections import deque # Queue를 사용하기 위한 import
import random # random value 생성
import atari_py # atrai 환경 사용을 위해서
import torch # pytorch version 0.4
import cv2  # Note that importing cv2 before torch may cause segfaults?
#왜 cv2를 torch 전에 import 하면 segment fault가 생겨?



class Env():

  def __init__(self, args):
    self.device = args.device # main.py 54번 줄 참고  args.device = torch.device('cuda') 단순히 GPU 확인차 받는 것
    self.ale = atari_py.ALEInterface() #아직 모름
    self.ale.setInt('random_seed', args.seed) #  #random seed value default is 123
    self.ale.setInt('max_num_frames', args.max_episode_length) #최대 episode 길이
    self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions [연속 동작 금지?]
    self.ale.setInt('frame_skip', 0)
    self.ale.setBool('color_averaging', False)  # 화면 이진화 color or gray
    self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options 내가 할 게임 불러다가 ROM에 담기
    actions = self.ale.getMinimalActionSet() #actionSet 가져오기
    self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions)) #action value를 dictionary로 짝지어줌
    self.lives = 0  # Life counter (used in DeepMind training)
    self.life_termination = False  # Used to check if resetting only from loss of life
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length) # 상태 담을 buffer 생성
    self.training = True  # Consistent with model training mode

  def _get_state(self):
    state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR) #input을 84x84의 흑백화면으로 받음
    return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255) # Tensor로 전환해서 return/  device는 gpu 쓰는지 안쓰는지

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(84, 84, device=self.device)) #버퍼 초기화



  def reset(self):
    '''

    Returns: torch.stack(list(self.state_buffer),0)

    '''
    if self.life_termination:
      self.life_termination = False  # Reset flag
      self.ale.act(0)  # Use a no-op after loss of life
    else:
      # Reset internals
      self._reset_buffer()
      self.ale.reset_game()
      # Perform up to 30 random no-ops before starting
      for _ in range(random.randrange(30)):
        self.ale.act(0)  # Assumes raw action 0 is always no-op
        if self.ale.game_over():
          self.ale.reset_game()
    # Process and return "initial" state
    observation = self._get_state()
    self.state_buffer.append(observation)
    self.lives = self.ale.lives()
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action):
    # Repeat action 4 times, max pool over last 2 frames
    frame_buffer = torch.zeros(2, 84, 84, device=self.device) #state 받을 frame_buffer 선언
    reward, done = 0, False
    for t in range(4):
      reward += self.ale.act(self.actions.get(action))
      if t == 2:
        frame_buffer[0] = self._get_state()
      elif t == 3:
        frame_buffer[1] = self._get_state()
      done = self.ale.game_over()
      if done:
        break
    observation = frame_buffer.max(0)[0] # observation 값 받기
    self.state_buffer.append(observation)
    # Detect loss of life as terminal in training mode
    if self.training:
      lives = self.ale.lives()
      if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
        self.life_termination = not done  # Only set flag when not truly done
        done = True
      self.lives = lives
    # Return state, reward, done
    return torch.stack(list(self.state_buffer), 0), reward, done  # return buffer, reward, done

  # Uses loss of life as terminal signal
  def train(self):
    self.training = True

  # Uses standard terminal signal
  def eval(self):
    self.training = False

  def action_space(self):
    return len(self.actions)

  def render(self):
    '''

    Returns: no return value
    it just show us the game screen.

    '''
    cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()
