#include<bits/stdc++.h>
using namespace std;
const int N = 1010;
int d , n , x[N] , y[N] , vis[N];
double dis(int i , int j){
    return sqrt((x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j]) * (y[i] - y[j]));   
}
void dfs(int id){
    vis[id] = 1;
    for(int i = 1 ; i <= n ; i ++) if(dis(i , id) <= d && !vis[i]) dfs(i);
}
int main() {
    cin >> n;
    for(int i = 1 ; i <= n ; i ++) cin >> x[i] >> y[i];
    cin >> d;
      dfs(1);
    for(int i = 1 ; i <= n ; i ++) cout << (vis[i] ? "1" : "0") << '\n';
    return 0;
}