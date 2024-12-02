//https://www.lanqiao.cn/problems/1181/learning/?page=2&first_category_id=1&second_category_id=8
#include <iostream>
using namespace std;
using ll = long long;

ll ksm(ll a,ll b,ll p){
  ll res=1;
  while(b){
    if(b&1) res=res*a%p;
    a=a*a%p;
    b>>=1;
  }
  return res;
}
int main()
{
  int t;
  cin>>t;
  while(t--){
    ll a,b,p;
    cin>>a>>b>>p;
    cout<<ksm(a,b,p)<<'\n';
  }
  // 请在此输入您的代码
  return 0;
}