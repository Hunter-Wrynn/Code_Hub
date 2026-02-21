class Solution {

    public:
        int minimumRecolors(string blocks, int k) {
            int cnt=0;
            int ans=5e3+1;
            for(int i=0;i<blocks.size();++i){
                if(blocks[i]=='W') cnt++;
                if(i-k+1<0) continue;
                ans=min(ans,cnt);
                if(blocks[i-k+1]=='W') cnt--;
            }
            return ans;
        }
    };