#include <bits/stdc++.h>

using namespace std;

const int N = 25;

int n, k;
string s, t;
int op[N], x[N], y[N];
vector<int> path;
bool st[N];
bool res;

void update()
{
    string str = s;
    for (int i = 0; i < path.size(); ++ i )
    {
        int j = path[i];
        int a = op[j], b = x[j], c = y[j];
        if (a == 1)
            str[b] = (str[b] - '0' + c) % 10 + '0';
        else
            swap(str[b], str[c]);
    }
    
    res |= str == t;
}

void dfs(int u)
{
    if (u == k + 1)
    {
        update();
        return;
    }
    
    for (int i = 1; i <= k; ++ i )
        if (!st[i])
        {
            st[i] = true;
            path.push_back(i);
            dfs(u + 1);
            path.pop_back();
            st[i] = false;
        }
    
    dfs(k + 1);
}

int main()
{
    cin >> n >> s >> t >> k;
    for (int i = 1; i <= k; ++ i )
        cin >> op[i] >> x[i] >> y[i];
    
    dfs(1);
    
    cout << (res? "Yes": "No") << endl;
    
    return 0;
}