from collections import deque

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_STRINGS = ['U', 'D', 'L', 'R']

def is_goal(state):
    return state == [1, 2, 3, 8, 0, 4, 7, 6, 5]

def print_board(board):
    for i in range(0, 9, 3):
        print(board[i:i+3])

def bfs(start):
    
    start_pos = start.index(0)
    
    queue = deque([(start, start_pos, "")])  
    visited = set()
    visited.add(tuple(start))

    while queue:
        board, zero_pos, path = queue.popleft()

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
                    queue.append((new_board, new_zero_pos, path + DIR_STRINGS[i]))

    print("No solution found.")

# 打印便利路径
def bfs_visit(start):
    start_pos = start.index(0)
    
    queue = deque([(start, start_pos, "")])  
    visited = set()
    visited_order = []  # 用来记录按进队列的顺序加入的状态
    visited.add(tuple(start))
    visited_order.append(start)

    while queue:
        board, zero_pos, path = queue.popleft()

        if is_goal(board):
            print("Solution found!")
            print("Moves:", path)
            print_board(board)
            break  
        
        zero_row, zero_col = divmod(zero_pos, 3)

        for i, (dx, dy) in enumerate(DIRECTIONS):
            new_row, new_col = zero_row + dx, zero_col + dy
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_zero_pos = new_row * 3 + new_col
                new_board = board[:]
                new_board[zero_pos], new_board[new_zero_pos] = new_board[new_zero_pos], new_board[zero_pos]

                if tuple(new_board) not in visited:
                    visited.add(tuple(new_board))
                    queue.append((new_board, new_zero_pos, path + DIR_STRINGS[i]))
                    visited_order.append(new_board)  

    print("Final visited set (in the order of queue):")
    
    for state in visited_order:
        print_board(state)
        print("")  

if __name__ == "__main__":
    start = [2, 0, 3, 1, 8, 4, 7, 6, 5] 
    bfs_visit(start)