#include <bits/stdc++.h>
using namespace std;
const int N = 15;
int a[N], n;
vector<int> v[N];

//cnt表示队伍数量，dfs返回cnt支队伍情况下是否成功分组
bool dfs(int cnt,int dep)
{
  if(dep == n + 1)
  {
    return true;
  }
  int num = a[dep];  
  for(int i = 1;i <= cnt; ++i)
  {
    bool tag = true;
    for(const auto &j : v[i])if(j % num == 0 || num % j == 0) tag = false;  
    if(!tag) continue;  
    v[i].push_back(num); 
    
    if(dfs(cnt, dep + 1)) return true; 
    
    v[i].pop_back();
  }
  return false;
}
int main()
{
  cin >> n;  for(int i = 1;i <= n; ++i) cin >> a[i];
  for(int i = 1;i <= 10; ++i)
  {
    if(dfs(i , 1))   
    {
      cout << i << '\n';
      break;
    }
  }
  return 0;
}