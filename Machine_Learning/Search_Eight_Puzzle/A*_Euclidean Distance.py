import heapq
import math


DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_STRINGS = ['U', 'D', 'L', 'R']


GOAL_STATE = [1, 2, 3, 8, 0, 4, 7, 6, 5]


def is_goal(state):
    return state == GOAL_STATE


def print_board(board):
    for i in range(0, 9, 3):
        print(board[i:i+3])


def euclidean_distance(state):
    distance = 0
    for i in range(9):
        if state[i] == 0:
            continue
        goal_pos = GOAL_STATE.index(state[i])
        
        current_row, current_col = divmod(i, 3)
        goal_row, goal_col = divmod(goal_pos, 3)
        
        distance += math.sqrt((current_row - goal_row)**2 + (current_col - goal_col)**2)
    return distance


def a_star(start):
    start_pos = start.index(0)  
    
    #f(n), g(n), state, zero_pos, path
    pq = []
    heapq.heappush(pq, (euclidean_distance(start), 0, start, start_pos, ""))
    

    visited = set()
    visited.add(tuple(start))

    while pq:
        f, g, board, zero_pos, path = heapq.heappop(pq)  

        if is_goal(board):  
            print("Solution found!")
            print("Moves:", path)
            print_board(board)
            return


        zero_row, zero_col = divmod(zero_pos, 3)


        for i, (dx, dy) in enumerate(DIRECTIONS):
            new_row, new_col = zero_row + dx, zero_col + dy
            if 0 <= new_row < 3 and 0 <= new_col < 3:  
                new_zero_pos = new_row * 3 + new_col
                new_board = board[:]

                new_board[zero_pos], new_board[new_zero_pos] = new_board[new_zero_pos], new_board[zero_pos]

                if tuple(new_board) not in visited:
                    visited.add(tuple(new_board))
                    new_g = g + 1  
                    new_h = euclidean_distance(new_board)  
              
                    heapq.heappush(pq, (new_g + new_h, new_g, new_board, new_zero_pos, path + DIR_STRINGS[i]))

    print("No solution found.")  

if __name__ == "__main__":
    start = [2, 0, 3, 1, 8, 4, 7, 6, 5]  
    a_star(start)
