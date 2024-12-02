//https://www.lanqiao.cn/problems/1225/learning/?page=3&first_category_id=1&second_category_id=8

#include <iostream>
#include <cstring>
#include <string>
using namespace std;
const int N=1e6+5;
char s[N];
int p[N];

int main()
{ 
  cin>>s+1;
  int n=strlen(s+1);
  for(int i=2*n+1;i>=1;i--){
    if(i&1) s[i]='#';
    else s[i]=s[i>>1];
  }
  s[0]='^';
  s[2*n+2]='$';
  int c=0,r=0;
  for(int i=1;i<=2*n+1;i++){
    p[i]=i<r?min(r-i,p[2*c-i]):1;
    while(s[i-p[i]]==s[i+p[i]]) p[i]++;
    if(i+p[i]>r) {
      c=i;

      r=i+p[i];
    }
  }
  int ans=0;
  for(int i=1;i<=2*n+1;i++){
    ans=max(ans,p[i]-1);
  }
  cout<<ans;
  // 请在此输入您的代码
  return 0;
}