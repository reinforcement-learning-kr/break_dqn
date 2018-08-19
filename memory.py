import random
from collections import namedtuple
import torch
Transition = namedtuple('Transitselion', ('timestep', 'state', 'action', 'reward', 'nonterminal')) 
blank_trans = Transition(0, torch.zeros(84, 84, dtype=torch.uint8), None, 0, False)
    """
	Output: Transition(timestep=0, state=tensor([[0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        ...,
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0]], dtype=torch.uint8), action=None, reward=0, nonterminal=False)
    """

# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
    """
	To store the experience and sample the data , we use Sum-Tree structure
	If we sort all samples according to their priorities and pick from left to right, it is a terrible efficiency.
	But if we use Sum-Tree, we don't need to sort array and save time to calculate.
    """

    def __init__(self, size):
    """ Initialization of Sum-Tree
        index =  The index of underlying data
        size  =  The size of underlying data
        full  =  If capacity is full, full is True 
        sum_tree = Total data including root and leaf node
        data  =   Underlying data
        max   =  maximum value

    Args:
        size: The size of underlying data
    """

        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.sum_tree = [0] * (2 * size - 1)  # Initialise fixed size tree with all (priority) zeros
        self.data = [None] * size  # Wrap-around cyclic buffer
        self.max = 1  # Initial max value to return (1 = 1^ω)

    # Propagates value up tree given a tree index
    def _propagate(self, index, value):
    """ It calculates Sum-Tree structure from leaf node to root node.
	Equation is
	parents = child[left]+child[right]
	It calculates untill parent index number is equal to 0 which means root node.
    Args:
        index: Tree index.
        value: Tree value at Tree index
    """

        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]	# parent = child[left] +right[right]
        if parent != 0:								# Untill it arrives at root node, it calculate forever
            self._propagate(parent, value)

    # Updates value given a tree index
    def update(self, index, value):
    """ Set the new data using suM_tree function
	Rrecalculate tree structure using _propagate function
	Updata max value using max function
    Args:
        index: Tree index 
        value: Tree value at Tree index
    """
        self.sum_tree[index] = value  # Set new value
        self._propagate(index, value)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
    """ Store underlying data in data structure
	To update value, we have to change from underlying index to total index.
	Equation is
	self.index + self.size -1
	Because total data is underlying data x2 -1
	And if all data is full, restore from 1 index 
    Args:
        data:  Underlying value
        value: Tree value

    """
        self.data[self.index] = data  # Store data in underlying data structure
        self.update(self.index + self.size - 1, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of a value in sum tree
    def _retrieve(self, index, value):
    """ Search the location using Sum-Tree structure

        Args:
            index: Tree index
	    value: Tree value at Tree index 

	To find underlying data location, compare left with right node value.
	If left is bigger, just choose left node using retrieve(left,value) function
	If right is bigger, choose right node and calculate (value - left node value).
	using retrieve(right, value-sum_tree[left]) function

	When left index number is bigger than the length of total tree,
	We stop _restrieve and return index.

        Returns: Location of a value in sum tree

    """

        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            return index
        elif value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.sum_tree[left])

    # Searches for a value in sum tree and returns value, data index and tree index
    def find(self, value):
    """ Searche the data index and tree index 
	We could know location of value using retrieve(0,value)
	Because 0 means root node, we could get location.

	Index is underlying data index, so we could calculate 
	Tree index - size +1 is equal to underlying data index

        Args:
            index: Tree index
            value: Tree value at Tree index 


        Returns: value at tree index, underlying data index, tree index

    """
        index = self._retrieve(0, value)  # Search for index of item from root
        data_index = index - self.size + 1
        return (self.sum_tree[index], data_index, index)  # Return value, data index, tree index

    # Returns data given a data index
    def get(self, data_index):
    """ Input value is index, return is value.
	if data_index is above size , get data from start.
        Args:
            data_index: underlying data index 

        Returns: underlying data value at data_index

    """
        return self.data[data_index % self.size]

    def total(self):
    """Total function means root node value 

        Returns: Root node value.

    """
        return self.sum_tree[0]


class ReplayMemory():
    def __init__(self, args, capacity):
        self.device = args.device
        self.capacity = capacity
        self.history = args.history_length
        self.discount = args.discount
        self.n = args.multi_step
        self.priority_weight = args.priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = args.priority_exponent
        self.t = 0  # Internal episode timestep counter
        self.transitions = SegmentTree(
            capacity)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities

    # Adds state and action at time t, reward and terminal at time t + 1

    def append(self, state, action, reward, terminal):
        """New transition with maximum priority
        The reason why we store new transition with maximum priority is
        to prevent no experience.

        If new transition is small transition,
        we always use high TD-error memory which make an overfitting

        Args:
            state: 84x84x4  (length of history is 4)
            action: stop left right
            reward: alive score
            terminal: dead
        """
        state = state[-1].mul(255).to(dtype=torch.uint8,
                                      device=torch.device('cpu'))  # Only store last frame and discretise to save memory
        self.transitions.append(Transition(self.t, state, action, reward, not terminal),
                                self.transitions.max)  # Store new transition with maximum priority
        self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0

    # Returns a transition with blank states where appropriate
    def _get_transition(self, idx):
        """

        Args:
            idx: index number

        Returns: The seriese of transition data

	Because history length is 4, it has the series of transition data.

	ex)
	If take a sample from memory, transition has the size of history(4)+ step(n) [n is multi-step size] 
	If terminal state, return is blank sample
        """
        transition = [None] * (self.history + self.n)
        transition[self.history - 1] = self.transitions.get(idx)
        for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
            if transition[t + 1].timestep == 0:
                transition[t] = blank_trans  # If future frame has timestep 0
            else:
                transition[t] = self.transitions.get(idx - self.history + 1 + t)
        for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
            if transition[t - 1].nonterminal:
                transition[t] = self.transitions.get(idx - self.history + 1 + t)
            else:
                transition[t] = blank_trans  # If prev (next) frame is terminal
        return transition

    # Returns a valid sample from a segment
    def _get_sample_from_segment(self, segment, i):
        """
	This function make a batch
        Args:
            segment: total_experience / batch_size
		i  : The order of batch.	
	In order to make batch, this function will be called 32 times.
	    return : i th element of batch

        """
        valid = False
        while not valid:
            sample = random.uniform(i * segment, (i + 1) * segment)  # Uniformly sample an element from within a segment
            prob, idx, tree_idx = self.transitions.find(
                sample)  # Retrieve sample from tree with un-normalised probability
            # Resample if transition straddled current index or probablity 0
            if (self.transitions.index - idx) % self.capacity > self.n and (
                    idx - self.transitions.index) % self.capacity >= self.history and prob != 0:
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0

        # Retrieve all required transition data (from t - h to t + n)
        transition = self._get_transition(idx)
        # Create un-discretised state and nth next state
        state = torch.stack([trans.state for trans in transition[:self.history]]).to(dtype=torch.float32,
                                                                                     device=self.device).div_(255)
        next_state = torch.stack([trans.state for trans in transition[self.n:self.n + self.history]]).to(
            dtype=torch.float32, device=self.device).div_(255)
        # Discrete action to be used as index
        action = torch.tensor([transition[self.history - 1].action], dtype=torch.int64, device=self.device)
        # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
        R = torch.tensor([sum(self.discount ** n * transition[self.history + n - 1].reward for n in range(self.n))],
                         dtype=torch.float32, device=self.device)
        # Mask for non-terminal nth next states
        nonterminal = torch.tensor([transition[self.history + self.n - 1].nonterminal], dtype=torch.float32,
                                   device=self.device)

        return prob, idx, tree_idx, state, action, R, next_state, nonterminal



    def sample(self, batch_size):
        """Important sampling weight

        Initial importance sampling weight	- B0 is 0.4
        Final importance sampling weight 	- Bf is 1

        Importance sampling weight value is annealed from 0.4 to 1 over course of training
        ,which will erase the biae of training


        For Example
        If Q-learning will be trained by large TD-error,
        Q-function will not be convergence. Because of large gradient magnitude.


        If Q-learning will be trained by small TD-error,
        Q-function will not be update.  Because lack of small TD-error experience and small gradient magnitude don't make Q-function update

        TD-error = R+(gamma)xQ(S,A) -Q(S',A')
        importance sampling(IS) = (N*P)^(-beta)/max(IS)
        MSE = (IS)*TD-error

        Args:
            batch_size: 32

        Returns: results of prioritized sampling

        """
        p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
        segment = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
        batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]  # Get batch of valid samples
        probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)
        states, next_states, = torch.stack(states), torch.stack(next_states)
        actions, returns, nonterminals = torch.cat(actions), torch.cat(returns), torch.stack(nonterminals)
        probs = torch.tensor(probs, dtype=torch.float32,
                             device=self.device) / p_total  # Calculate normalised probabilities
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
        weights = weights / weights.max()  # Normalise by max importance-sampling weight from batch
        return tree_idxs, states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs, priorities):
        """Stochastic sampling method

        Priority_exponent value is 0.5
        A stochastic sampling method interpolates between pure greedy prioritization and uniform random sampling


        P(i) = p(i)^alpha  / sum( p(i)^(alpha))

        For Example
        If priority_exponent(alpha) is 0, experience be choosed by uniform random sampling
        If priority_exponent(alpha) is 1, experience be choosed by pure greedy prioritization
        so 0.5 is suitable priority exponent value to sample

        Args:
            idxs: Total number of transition
            priorities: prioritization

	In per, There are two type of prioritization.
	First is proportional prioritization where p = TD-error +epsilon 
	The reason of plus epsilon is to prevent zero

	Second is rank-based prioritization where p = 1/rank(index)
	In prioritized experience replay paper, rank-based prioritization is more robust than prioritization


        """
        priorities.pow_(self.priority_exponent)
        [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

    # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration
        # Create stack of states
        state_stack = [None] * self.history
        state_stack[-1] = self.transitions.data[self.current_idx].state
        prev_timestep = self.transitions.data[self.current_idx].timestep
        for t in reversed(range(self.history - 1)):
            if prev_timestep == 0:
                state_stack[t] = blank_trans.state  # If future frame has timestep 0
            else:
                state_stack[t] = self.transitions.data[self.current_idx + t - self.history + 1].state
                prev_timestep -= 1
        state = torch.stack(state_stack, 0).to(dtype=torch.float32, device=self.device).div_(
            255)  # Agent will turn into batch
        self.current_idx += 1
        return state
