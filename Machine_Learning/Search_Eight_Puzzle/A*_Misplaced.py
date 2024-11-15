import heapq


DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_STRINGS = ['U', 'D', 'L', 'R']


GOAL_STATE = [1, 2, 3, 8, 0, 4, 7, 6, 5]

def is_goal(state):
    return state == GOAL_STATE

def print_board(board):
    for i in range(0, 9, 3):
        print(board[i:i+3])

def misplaced_tiles(state):
    distance = 0
    for i in range(9):
        if state[i] != 0 and state[i] != GOAL_STATE[i]:
            distance += 1
    return distance

def a_star(start):
    start_pos = start.index(0) 
    
    pq = []
    heapq.heappush(pq, (misplaced_tiles(start), 0, start, start_pos, ""))
    
    # 记录已访问的状态
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
                    new_h = misplaced_tiles(new_board)  
                    
                    heapq.heappush(pq, (new_g + new_h, new_g, new_board, new_zero_pos, path + DIR_STRINGS[i]))

    print("No solution found.") 

if __name__ == "__main__":
    start = [2, 0, 3, 1, 8, 4, 7, 6, 5]  
    a_star(start)
