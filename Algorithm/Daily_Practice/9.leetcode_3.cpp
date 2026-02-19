class Solution {
    public:
        int lengthOfLongestSubstring(string s) {
            unordered_set<char> occ;
            int n=s.size();
            int r=-1,ans=0;
            for(int i=0;i<n;++i){
                if(i!=0){
                    occ.erase(s[i-1]);
                }
                while(r+1<n&&!occ.count(s[r+1])){
                    occ.insert(s[r+1]);
                    r++;
                }
                ans=max(ans,r-i+1);
            }
            return ans;
        }
    };