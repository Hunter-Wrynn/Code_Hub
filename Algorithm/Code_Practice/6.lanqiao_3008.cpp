#include<bits/stdc++.h>
using namespace std;
const int N=1e6+3;
int cnt[N],prefix[N];

void dfs(int dep,int mul,int sum,int st){
  if(mul>1e6) return;
  if(dep==4){
    cnt[mul]++;
    return;
  }
  int up=pow(1e6/mul,1.0/(4-dep))+3;

  for(int i=st+1;i<(dep==3?min(sum,up):up);++i){
    dfs(dep+1,mul*i,sum+i,i);
  }
}
int main()
{
  dfs(1,1,0,0);
  for(int i=1;i<=1e6;++i) prefix[i]+=prefix[i-1]+cnt[i];
  int n;
  cin>>n;
  while(n--){
    int a,b;
    cin>>a>>b;
    cout<<prefix[b]-prefix[a-1];
    cout<<'\n';
  }
  return 0;
}