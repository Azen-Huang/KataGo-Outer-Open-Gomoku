#include <math.h>
#include <iostream>
#include <iomanip>
#include <gomoku.h>

Gomoku::Gomoku(unsigned int n, unsigned int n_in_row, int first_color)
    : n(n), n_in_row(n_in_row), cur_color(first_color), last_move(-1), sz(0) {
  this->board = std::vector<std::vector<int>>(n, std::vector<int>(n, 0));
}



std::vector<int> Gomoku::get_legal_moves() {
  auto n = this->n;
  std::vector<int> legal_moves(this->get_action_size(), 0);
  if (this->sz == 0) {
    const std::vector<int> first_step_legal_moves = {
      16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,
      16,  31,  46,  61,  76,  91,  106, 121, 136, 151, 166, 181, 196,
      196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208,
      28,  43,  58,  73,  88,  103, 118, 133, 148, 163, 178, 193, 208,
    };
    for (auto& legal_move : first_step_legal_moves) {
      legal_moves[legal_move] = 1;
    }
    return legal_moves;
  }

  
  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      if (this->board[i][j] == 0) {
        legal_moves[i * n + j] = 1;
      }
    }
  }

  return legal_moves;
}

bool Gomoku::has_legal_moves() {
  auto n = this->n;

  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      if (this->board[i][j] == 0) {
        return true;
      }
    }
  }
  return false;
}

void Gomoku::execute_move(move_type move) {
  auto i = move / this->n;
  auto j = move % this->n;

  if (!this->board[i][j] == 0) {
    throw std::runtime_error("execute_move borad[i][j] != 0.");
  }

  this->board[i][j] = this->cur_color;
  this->last_move = move;
  // change player
  this->cur_color = -this->cur_color;
  this->sz += 1;
}

std::vector<int> Gomoku::get_game_status() {
  // return (is ended, winner)
  auto n = this->n;
  auto n_in_row = this->n_in_row;

  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      if (this->board[i][j] == 0) {
        continue;
      }

      if (j <= n - n_in_row) {
        auto sum = 0;
        for (unsigned int k = 0; k < n_in_row; k++) {
          sum += this->board[i][j + k];
        }
        if (abs(sum) == n_in_row) {
          return {1, this->board[i][j]};
        }
      }

      if (i <= n - n_in_row) {
        auto sum = 0;
        for (unsigned int k = 0; k < n_in_row; k++) {
          sum += this->board[i + k][j];
        }
        if (abs(sum) == n_in_row) {
          return {1, this->board[i][j]};
        }
      }

      if (i <= n - n_in_row && j <= n - n_in_row) {
        auto sum = 0;
        for (unsigned int k = 0; k < n_in_row; k++) {
          sum += this->board[i + k][j + k];
        }
        if (abs(sum) == n_in_row) {
          return {1, this->board[i][j]};
        }
      }

      if (i <= n - n_in_row && j >= n_in_row - 1) {
        auto sum = 0;
        for (unsigned int k = 0; k < n_in_row; k++) {
          sum += this->board[i + k][j - k];
        }
        if (abs(sum) == n_in_row) {
          return {1, this->board[i][j]};
        }
      }
    }
  }

  if (this->has_legal_moves()) {
    return {0, 0};
  } else {
    return {1, 0};
  }
}

void Gomoku::display() const {
  auto n = this->board.size();
  std::cout << "   ";
  for(char i = 'A'; i <= 'O'; ++i){
      std::cout << i << " ";
  }
  std::cout << std::endl;
  std::cout << "   ";
  for(int i = 'A'; i <= 'O'; ++i){
      std::cout << "--";
  }
  std::cout << std::endl;
  std::cout << 15 << "|";
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      char stone;
      if (this->board[i][j] == 0) {
        stone = '-';
      }
      else if (this->board[i][j] == 1) {
        stone = 'o';
      }
      else {
        stone = 'x';
      }
      std::cout << stone << " ";
    }
    std::cout << std::endl;
    if(i != 14) {
        std::cout << std::setw(2) << 15 - i - 1 << "|";
    }
  }
  std::cout << std::endl << std::endl;
}