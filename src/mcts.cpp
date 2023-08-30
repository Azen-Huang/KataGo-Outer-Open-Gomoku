#include <math.h>
#include <float.h>
#include <numeric>
#include <iostream>
#include <random>
#include <mcts.h>
#include <dirichlet.h>

template <typename T>
void print_policy(T& vec){
    std::cout << "   ";
    for(char i = 'A'; i <= 'O'; ++i){
        std::cout << " " << i << "  ";
    }
    std::cout << std::endl;
    std::cout << "   ";
    for(int i = 'A'; i <= 'O'; ++i){
        std::cout << " -- ";
    }
    std::cout << std::endl;
    std::cout << 15 << "|";
    for(int i = 1; i <= 225; ++i){
        if(vec[i - 1] >= 1){
            std::cout << std::setw(3) << std::setfill('0') << vec[i - 1] << " ";
        }
        else{
            std::cout << " - " << " ";
        }
        if(i % 15 == 0){
            std::cout << std::endl;
            if(i != 225)
                std::cout << std::setw(2) << (225 - i) / 15 << "|";
        }
    }
    return;
}

// TreeNode
TreeNode::TreeNode()
    : parent(nullptr),
      is_leaf(true),
      virtual_loss(0),
      n_visited(0),
      p_sa(0),
      q_sa(0),
      level(0) {}

TreeNode::TreeNode(TreeNode *parent, double p_sa, unsigned int action_size, int level)
    : parent(parent),
      children(action_size, nullptr),
      is_leaf(true),
      virtual_loss(0),
      n_visited(0),
      q_sa(0),
      p_sa(p_sa),
      level(level) {}

TreeNode::TreeNode(
    const TreeNode &node) {  // because automic<>, define copy function
  // struct
  this->parent = node.parent;
  this->children = node.children;
  this->is_leaf = node.is_leaf;

  this->n_visited.store(node.n_visited.load());
  this->p_sa = node.p_sa;
  this->q_sa = node.q_sa;

  this->virtual_loss.store(node.virtual_loss.load());
  this->level = node.level;
}

TreeNode &TreeNode::operator=(const TreeNode &node) {
  if (this == &node) {
    return *this;
  }

  // struct
  this->parent = node.parent;
  this->children = node.children;
  this->is_leaf = node.is_leaf;

  this->n_visited.store(node.n_visited.load());
  this->p_sa = node.p_sa;
  this->q_sa = node.q_sa;
  this->virtual_loss.store(node.virtual_loss.load());

  this->level = node.level;
  return *this;
}

unsigned int TreeNode::select(double c_puct, double c_virtual_loss) {
  double best_value = -DBL_MAX;
  unsigned int best_move = 0;
  TreeNode *best_node;
  
  if (this->level == 0) {
    int best_node_visted = INT_MAX;
    const unsigned int sum_n_visited = this->n_visited.load() + 1;
    for (unsigned int i = 0; i < this->children.size(); i++) {
      // empty node
      if (children[i] == nullptr) {
        continue;
      }

      int n_forced = int(sqrt(2.0 * children[i]->p_sa * (double)(sum_n_visited + (1e-8))));

      if (n_forced > children[i]->n_visited.load()) {
        if (children[i]->n_visited.load() < best_node_visted) {
          children[i]->virtual_loss++;
          return i;
        }
      }
    }
  }
  
  double total_visited_policy = 0.0;
  for (unsigned int i = 0; i < this->children.size(); i++) {
    // empty node
    if (children[i] == nullptr) {
      continue;
    }
    if (children[i]->n_visited.load() > 0) {
      total_visited_policy += children[i]->p_sa;
    }
  }

  for (unsigned int i = 0; i < this->children.size(); i++) {
    // empty node
    if (children[i] == nullptr) {
      continue;
    }
    unsigned int sum_n_visited = this->n_visited.load() + 1;
    double cur_value = children[i]->get_value(c_puct, c_virtual_loss, sum_n_visited, this->q_sa, total_visited_policy);
    if (cur_value > best_value) {
      best_value = cur_value;
      best_move = i;
      best_node = children[i];
    }
  }
  

  // add vitural loss
  best_node->virtual_loss++;

  return best_move;
}

void TreeNode::expand(const std::vector<double> &action_priors) {
  {
    // get lock
    std::lock_guard<std::mutex> lock(this->lock);

    if (this->is_leaf) {
      unsigned int action_size = this->children.size();

      for (unsigned int i = 0; i < action_size; i++) {
        // illegal action
        if (abs(action_priors[i] - 0) < FLT_EPSILON) {
          continue;
        }
        this->children[i] = new TreeNode(this, action_priors[i], action_size, this->level + 1);
      }

      // not leaf
      this->is_leaf = false;
    }
  }
}

void TreeNode::backup(double value) {
  // If it is not root, this node's parent should be updated first
  if (this->parent != nullptr) {
    this->parent->backup(-value);
  }

  // remove vitural loss
  this->virtual_loss--;

  // update n_visited
  unsigned int n_visited = this->n_visited.load();
  this->n_visited++;

  // update q_sa
  {
    std::lock_guard<std::mutex> lock(this->lock);
    this->q_sa = (n_visited * this->q_sa + value) / (n_visited + 1);
  }
}

double TreeNode::get_value(double c_puct, double c_virtual_loss,
                           unsigned int sum_n_visited, double parent_q_sa, double total_visited_policy) const {
  // u
  auto n_visited = this->n_visited.load();
  double u = (c_puct * this->p_sa * sqrt(sum_n_visited) / (1 + n_visited));

  // virtual loss
  double virtual_loss = c_virtual_loss * this->virtual_loss.load();
  // int n_visited_with_loss = n_visited - virtual_loss;
  
  if (n_visited <= 0) {
    auto cfg_fpu_reduction = 0.25f;
    auto fpu_reduction = cfg_fpu_reduction * std::sqrt(total_visited_policy);
    return u + parent_q_sa - fpu_reduction;
  } else {
    return u + (this->q_sa * n_visited - virtual_loss) / n_visited;
  }
}

// MCTS
MCTS::MCTS(NeuralNetwork *neural_network, unsigned int thread_num, double c_puct,
           unsigned int num_mcts_sims, double c_virtual_loss,
           unsigned int action_size, bool add_noise)
    : neural_network(neural_network),
      thread_pool(new ThreadPool(thread_num)),
      thread_num(thread_num),
      c_puct(c_puct),
      num_mcts_sims(num_mcts_sims),
      c_virtual_loss(c_virtual_loss),
      action_size(action_size),
      add_noise(add_noise),
      root(new TreeNode(nullptr, 1., action_size, 0), MCTS::tree_deleter){}

void MCTS::update_with_move(int last_action) {
  auto old_root = this->root.get();

  // reuse the child tree
  if (last_action >= 0 && old_root->children[last_action] != nullptr) {
    // unlink
    TreeNode *new_node = old_root->children[last_action];
    old_root->children[last_action] = nullptr;
    new_node->parent = nullptr;
    new_node->level = 0;
    this->root.reset(new_node);
  } else {
    this->root.reset(new TreeNode(nullptr, 1., this->action_size, 0));
  }
}

void MCTS::tree_deleter(TreeNode *t) {
  if (t == nullptr) {
    return;
  }

  // remove children
  for (unsigned int i = 0; i < t->children.size(); i++) {
    if (t->children[i]) {
      tree_deleter(t->children[i]);
    }
  }

  // remove self
  delete t;
}

std::vector<double> MCTS::get_action_probs(Gomoku *gomoku, double temp, bool is_debug) {
  // submit simulate tasks to thread_pool
  std::vector<std::future<void>> futures;
  // gomoku->display();
  for (unsigned int i = 0; i < this->num_mcts_sims; i++) {
    // copy gomoku
    auto game = std::make_shared<Gomoku>(*gomoku);
    auto future =
        this->thread_pool->commit(std::bind(&MCTS::simulate, this, game));

    // future can't copy
    futures.emplace_back(std::move(future));
  }

  // wait simulate
  for (unsigned int i = 0; i < futures.size(); i++) {
    futures[i].wait();
  }

  // calculate probs
  std::vector<double> action_probs(gomoku->get_action_size(), 0);
  std::vector<double> action_value(gomoku->get_action_size(), -2.0); // python check

  const auto &children = this->root->children;

  double sum = 0;
  unsigned int sum_n_visited = this->root->n_visited.load() + 1;
  std::vector<int> children_visited(225, 0);
  unsigned int max_n_visited = 0;
  for (unsigned int i = 0; i < children.size(); i++) {
    if (children[i]) {
      unsigned int n_visited = children[i]->n_visited.load();
      if (max_n_visited < n_visited) {
        max_n_visited = n_visited;
      }
    }
  }
  
  for (unsigned int i = 0; i < children.size(); i++) {
    if (children[i]) {
      action_value[i] = children[i]->q_sa;
      // int n_forced = ceil(sqrt(2.0 * children[i]->p_sa * (double)(sum_n_visited + (1e-8)))) + 2;
      int n_forced = int(sqrt(2.0 * children[i]->p_sa * (double)(sum_n_visited + (1e-8)))) + (this->thread_num);
      int n_visited = children[i]->n_visited.load();
      if (n_visited != max_n_visited) {
        n_visited -= n_forced;
      }
      
      if (n_visited > 0) {
        children_visited[i] = n_visited;
        if (temp - 1e-3 < FLT_EPSILON) {
          action_probs[i] = double(n_visited);
        }
        else {
          action_probs[i] = pow(n_visited, 1 / temp);
        }
        sum += action_probs[i];
      }
    }
  }

  // if (children[i] && children[i]->n_visited.load() > 0) {
  //   action_probs[i] = pow(children[i]->n_visited.load(), 1 / temp);
  //   sum += action_probs[i];
  // }

  if (is_debug) {
    gomoku->display();
    print_policy(children_visited); //
  }
  // std::cout << std::endl << std::endl; //

  // renormalization
  std::for_each(action_probs.begin(), action_probs.end(),
                [sum](double &x) { x /= sum; });
  gomoku->update_action_value(action_value);
  return action_probs;
}

void MCTS::simulate(std::shared_ptr<Gomoku> game) {
  // execute one simulation
  auto node = this->root.get();
  this->c_puct = 1.25 + 1.0 * log2((game->get_size() + 1.0 + 19652.0) / 19652.0);

  //------------------------------------
  while (true) {
    if (node->get_is_leaf()) {
      break;
    }

    // select
    auto action = node->select(this->c_puct, this->c_virtual_loss);
    game->execute_move(action);
    node = node->children[action];
  }

  // get game status
  auto status = game->get_game_status();
  double value = 0;

  // not end
  if (status[0] == 0) {
    // predict action_probs and value by neural network
    std::vector<double> action_priors(this->action_size, 0);

    auto future = this->neural_network->commit(game.get());
    auto result = future.get();

    action_priors = std::move(result[0]);
    value = result[1][0];
    // mask invalid actions
    auto legal_moves = game->get_legal_moves();

    // add dirichlet noise
    if (node->level == 0 && add_noise) {
      
      // set alpha
      double legal_moves_size = std::reduce(legal_moves.begin(), legal_moves.end(), 0.0);
      double a = 10.0 / legal_moves_size;
      std::vector<double> alpha(225, a);
      // initialize rng
      std::random_device rd;
      std::mt19937 gen(rd());
      // Dirichlet distribution using mt19937 rng
      dirichlet_distribution<std::mt19937> d(alpha);
      std::vector<double> dirichlet_noise = d(gen);
      for (int i = 0; i < this->action_size; ++i) {
        action_priors[i] = 0.75 * action_priors[i] + 0.25 * dirichlet_noise[i];
      }
    }

    double sum = 0;
    for (unsigned int i = 0; i < action_priors.size(); i++) {
      if (legal_moves[i] == 1) {
        sum += action_priors[i];
      } else {
        action_priors[i] = 0;
      }
    }

    // renormalization
    if (sum > FLT_EPSILON) {
      std::for_each(action_priors.begin(), action_priors.end(),
                    [sum](double &x) { x /= sum; });
    } else {
      // all masked

      // NB! All valid moves may be masked if either your NNet architecture is
      // insufficient or you've get overfitting or something else. If you have
      // got dozens or hundreds of these messages you should pay attention to
      // your NNet and/or training process.
      std::cout << "All valid moves were masked, do workaround." << std::endl;

      sum = std::accumulate(legal_moves.begin(), legal_moves.end(), 0);
      for (unsigned int i = 0; i < action_priors.size(); i++) {
        action_priors[i] = legal_moves[i] / sum;
      }
    }

    // expand
    node->expand(action_priors);

  } else {
    // end
    auto winner = status[1];
    value = (winner == 0 ? 0 : (winner == game->get_current_color() ? 1 : -1));
  }

  // value(parent -> node) = -value
  node->backup(-value);
  return;
}
