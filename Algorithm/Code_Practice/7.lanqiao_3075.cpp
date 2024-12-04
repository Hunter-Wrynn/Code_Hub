//https://www.lanqiao.cn/problems/3075/learning/?page=1&first_category_id=1&problem_id=3075

#include <bits/stdc++.h>
using namespace std;
const int N = 1e5;
vector<int> e;
int ans[N + 1],n,t,pre[N + 1];
bool check(int sum) {
  for(int i = 0; i < e.size(); i ++) {
    int em = e[i];
    if(sum <= 2 * em) return false;
  }
  return true;
}
void dfs(int d,int last, int mul, int sum) {

  if(d >= n) {
    if(check(sum)) ans[mul] ++;            
    return ;
  }

  for(int i = last + 1; ; i ++) {
    if(mul * pow(i,n - d) > N) break;
    e.push_back(i);
    dfs(d + 1, i, mul * i, sum + i);
    e.pop_back();
  }

}
int main() {
  cin>>t>>n;
  dfs(0,0,1,0);
  for(int i = 1; i <= N; i ++) pre[i] = pre[i - 1] + ans[i];
  while(t --) {
    int l,r;
    cin>>l>>r;
    cout<<pre[r] - pre[l - 1]<<endl;
  }
}