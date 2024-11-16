#include<bit/stdc++.h>
using namespace std;

void dfs(int current, int n) {
    if (current > n) return;
    std::cout << current << " "; // 输出当前数字
    for (int i = 0; i <= 9; ++i) {
        int next = current * 10 + i;
        if (next <= n) {
            dfs(next, n); // 递归调用
        }
    }
}

int main() {
    int n;
    std::cin >> n;
    for (int i = 1; i <= 9; ++i) {
        dfs(i, n); // 从每个数字开始
    }
    return 0;
}
