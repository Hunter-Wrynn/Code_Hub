//https://www.lanqiao.cn/problems/178/learning/?page=1&first_category_id=1&problem_id=178


#include <bits/stdc++.h>

const int N=1e3+5;
char mp[N][N];
int n,scc,col[N][N];
bool vis[N*N];
int dx[]={0,0,-1,1};
int dy[]={1,-1,0,0};

using namespace std;
void dfs(int x,int y){
  col[x][y]=scc; 
  for(int i=0;i<4;i++){
    int nx=x+dx[i];
    int ny=y+dy[i];
    if(col[nx][ny]||mp[nx][ny]=='.') continue;

    dfs(nx,ny);

  }
}
int main()
{
  ios::sync_with_stdio(0),cin.tie(0),cout.tie(0);
  cin>>n;
  for(int i=1;i<=n;i++){
    cin>>mp[i]+1;
  }
  for(int i=1;i<=n;i++){
    for(int j=1;j<=n;j++){
      if(col[i][j]||mp[i][j]=='.') continue;

      scc++;
      dfs(i,j);
    }
  }
  int ans=0;
  for(int i=1;i<=n;i++){
    for(int j=1;j<=n;j++){
      if (mp[i][j]=='.') continue;
      bool tag=true;
      for(int k=0;k<4;k++){

        int x=i+dx[k];
        int y=j+dy[k];
        if(mp[x][y]=='.'){
          tag =false;
        }
      }
      if(tag){
        if(!vis[col[i][j]]) ans++;
        vis[col[i][j]]=true;
      }
    }
  }
  cout<<scc-ans;
  // 请在此输入您的代码
  return 0;
}