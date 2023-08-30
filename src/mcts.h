#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include <thread>
#include <atomic>

#include <gomoku.h>
#include <thread_pool.h>
#include <libtorch.h>

class TreeNode {
 public:
  // friend class can access private variables
  friend class MCTS;

  TreeNode();
  TreeNode(const TreeNode &node);
  TreeNode(TreeNode *parent, double p_sa, unsigned action_size, int level);

  TreeNode &operator=(const TreeNode &p);

  unsigned int select(double c_puct, double c_virtual_loss);
  void expand(const std::vector<double> &action_priors);
  void backup(double leaf_value);

  double get_value(double c_puct, double c_virtual_loss,
                   unsigned int sum_n_visited, double parent_q_sa, double total_visited_policy) const;
  inline bool get_is_leaf() const { return this->is_leaf; }

 private:
  // store tree
  TreeNode *parent;
  std::vector<TreeNode *> children;
  bool is_leaf;
  std::mutex lock;

  std::atomic<unsigned int> n_visited;
  double p_sa;
  double q_sa;
  std::atomic<int> virtual_loss;
  int level;
};

class MCTS {
 public:
  MCTS(NeuralNetwork *neural_network, unsigned int thread_num, double c_puct,
       unsigned int num_mcts_sims, double c_virtual_loss,
       unsigned int action_size, bool add_noise);
  std::vector<double> get_action_probs(Gomoku *gomoku, double temp = 1e-3, bool is_debug = false);
  void update_with_move(int last_move);

 private:
  void simulate(std::shared_ptr<Gomoku> game);
  static void tree_deleter(TreeNode *t);

  // variables
  std::unique_ptr<TreeNode, decltype(MCTS::tree_deleter) *> root;
  std::unique_ptr<ThreadPool> thread_pool;
  NeuralNetwork *neural_network;

  unsigned int action_size;
  unsigned int num_mcts_sims;
  unsigned int thread_num;
  double c_puct;
  double c_virtual_loss;
  bool add_noise;
};
