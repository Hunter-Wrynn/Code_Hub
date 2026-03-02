class Solution {
public:
    int minFlips(string s) {
        int cnt=0;
        int n=s.size();
        int ans=n;
        for(int i=0;i<2*n-1;++i){
            if(s[i%n]%2!=i%2) cnt++;
            if(i-n+1<0) continue;
            ans=min({cnt,ans,n-cnt});
            if(s[i-n+1]%2!=(i-n+1)%2) cnt--;
        }
        return ans;
    }
};

