from collections import deque

DIRECTIONS = [ (1, 0), (0, -1), (0, 1),(-1, 0)]
DIR_STRINGS = ['U', 'D', 'L', 'R']

def is_goal(state):
    return state == [1, 2, 3, 8, 0, 4, 7, 6, 5]

def print_board(board):
    for i in range(0, 9, 3):
        print(board[i:i+3])

def dfs(start):
    start_pos = start.index(0)
    
    # 使用栈来模拟DFS
    stack = [(start, start_pos, "")]  
    visited = set()
    visited.add(tuple(start))

    while stack:
        board, zero_pos, path = stack.pop()  # 弹出栈顶的状态

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
                    stack.append((new_board, new_zero_pos, path + DIR_STRINGS[i]))  # 将新状态压入栈

    print("No solution found.")

def dfs_visit(start):
    start_pos = start.index(0)
    
    # 使用栈来模拟DFS
    stack = [(start, start_pos, "")]  
    visited = set()
    visited.add(tuple(start))

    visited_order = []  # 用来记录按进队列的顺序加入的状态

    while stack:
        board, zero_pos, path = stack.pop()  # 弹出栈顶的状态

        if len(visited_order) < 5:  # 只打印前 5 个状态
            visited_order.append(board)
        
        if is_goal(board):
            print("Solution found!")
            print("Moves:", path)
            print_board(board)
            print("First 5 visited states:")
            # 打印前 5 个状态
            for state in visited_order:
                print_board(state)
                print("")  # 空行分隔每个状态
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
                    stack.append((new_board, new_zero_pos, path + DIR_STRINGS[i]))  # 将新状态压入栈

    print("No solution found.")




if __name__ == "__main__":
    start = [2, 0, 3, 1, 8, 4, 7, 6, 5]  # 输入的初始状态
    dfs_visit(start)
