#include <bits/stdc++.h>
const int N=1e5+3;
int a[N];
int t[N];
int n,m;

using namespace std;

int lowbit(int x){
  return x&-x;
}
void update(int k,int x){
  a[k]+=x;
  for(int i=k;i<=n;i+=lowbit(i)){
    t[i]+=x;
  }
}
int pre(int k){
  int sum=0;
  for(int i=k;i>0;i-=lowbit(i)){
    sum+=t[i];
  }
  return sum;
}

int getsum(int l,int r){
  return pre(r)-pre(l-1);
}

int main()
{
  cin>>n>>m;
  for(int i=1;i<=n;i++){
    int x;
    cin>>x;
    update(i,x);
  }
  while(m--){
    int op;
    cin>>op;
    if(op==1){
      int i,x;
      cin>>i>>x;
      update(i,x-a[i]);
    }
    else{
      int i;
      cin>>i;
      cout<<b(i)<<'\n';
    }
  }

  return 0;
}