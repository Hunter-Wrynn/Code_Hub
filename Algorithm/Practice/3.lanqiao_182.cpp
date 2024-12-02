//https://www.lanqiao.cn/problems/182/learning/?page=1&first_category_id=1&problem_id=182

#include<bits/stdc++.h>
using namespace std;
const int N=1e5+4;
int a[N],v[N];
int n;
int t;
int max=0;

void dfs(int x,int y){
    if(v[x]){
        if(a[x]==a[t]){
            if(y>max){
                max=y;
            }
        }
        return;
    }
    else{
        v[x]=1;
        dfs(a[x],y+1);
        v[x]=0;
    }
}
int main(){


    cin>>n;
    for(int i=1;i<=n;++i){
        cin>>a[i];
    }
    for(int i=1;i<=n;++i){
        t=i;
        dfs(i,0);
    }
}