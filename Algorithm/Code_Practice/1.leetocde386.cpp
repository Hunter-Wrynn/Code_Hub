#include<bits/stdc++.h>
using namespace std;

void dfs(int current, int n) {
    if (current > n) return;
    cout << current << " "; 
    for (int i = 0; i <= 9; ++i) {
        int next = current * 10 + i;
        if (next <= n) {
            dfs(next, n); 
        }
    }
}

int main() {
    int n;
    cin >> n;
    for (int i = 1; i <= 9; ++i) {
        dfs(i, n); 
    }
    return 0;
}
