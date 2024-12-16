#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;
const int N = 1010;
typedef long long LL;
int n, m;  
int a[N][N];  
bool st[N][N];  
const int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1}; 


void dfs(int x, int y, LL &s)
{
    s+=1ll*a[x][y];  
    st[x][y]=true;
   
    for(int i=0; i<4; i++)
    {
        int nx=x+dx[i], ny=y+dy[i];
       
        if(nx<0 || nx>=n || ny<0 || ny>=m || st[nx][ny] || !a[nx][ny]) continue;  
        dfs(nx,ny,s);
    }
}

int main()
{
    cin>>n>>m;
    LL res1=0; 

    for(int i=0; i<n; i++) 
        for(int j=0; j<m; j++)
            cin>>a[i][j];
    memset(st, 0, sizeof st);  
    for(int i=0; i<n; i++)
        for(int j=0; j<m; j++)
            if(a[i][j] && !st[i][j])  
            {
                LL res=0;
                dfs(i,j,res);
                res1=max(res1,res);  
            }
    cout<<res1<<'\n';
    return 0;
}