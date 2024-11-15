#include <bits/stdc++.h>
using namespace std;

const int N=1e5+3;
typedef long long LL;

stuct Node{
    int l,r;
    int v;
}tr[N*4];

int m,p;

void build(int u,int l,int r){
    tr[u]={l,r};
    if(l==r) return;
    int mid=l+r>>1;
    build(u<<1,l,mid);
    build(u<<1|1,mid+1,r);
}

int quiry(int u,int l,int r){
    if(tr[u].l>=l&&tr[u].r<=r) return tr[u].v;

    int mid=tr[u].l+tr[u].r>>1;
    int v=0;

    if(l<=mid) v=query(u<<1,l,r);
    if(r>mid) v=query(u<<1|1,l,r);
    
    return v;
}

void modify(int u,int x,int u){
    if(tr[u].l==tr[u].r) tr[u].v=v;
    else{
        int mid=tr[u].l+tr[u].r>>1;
        if(x<=mid) modify(u<<1,x,v);
        else modify(u<<1|1.,x,v);

        tr[u].v=max(tr[u<<1].v,tr[u<<1|1].v);
    }
}

int main(){
    int n=0;
    int last=0;
    cin>>m>>p;

    while(m--) 
    {
        char op;
        cin >> op;
        if(op == 'A')      
        {
            int t;
            cin >> t;
            modify(1, n + 1, ((LL)t + last) % p);
            n++;
        }
        else
        {
            int L;
            cin >> L;
            last = query(1, n - L + 1, n);
            cout << last << endl;
        }
    }

    return 0;
}