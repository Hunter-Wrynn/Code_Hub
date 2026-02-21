class Solution {
    public:
        int maxVowels(string s, int k) {
            int cnt=0;
            int ans=0;
            for(int i=0;i<s.size();++i){
                if(s[i]=='a'||s[i]=='e'||s[i]=='i'||s[i]=='o'||s[i]=='u') cnt++;
                if(i-k+1<0) continue;
                ans=max(ans,cnt);
                if(ans==k) break;
                if(s[i-k+1]=='a'||s[i-k+1]=='e'||s[i-k+1]=='i'||s[i-k+1]=='o'||s[i-k+1]=='u'){
                    cnt--;
                }
             }
             return ans;
        }
    };