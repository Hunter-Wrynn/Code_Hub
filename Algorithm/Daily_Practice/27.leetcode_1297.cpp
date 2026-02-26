// 脑筋急转弯：只需考虑 minSize，maxSize 无效（长度为 minSize 的子串出现次数 >= 任一更长子串）
class Solution {
    public:
        int maxFreq(string s, int maxLetters, int minSize, int maxSize) {
            int cnt=0;
            int kind=0;
            int ans=0;
            unordered_map<string,int> mp;
            int ccnt[26]{};
            for(int i=0;i<s.size();++i){
                int a=s[i]-'a';
                if(ccnt[a]==0) kind++;
                ccnt[a]++;
                int k=minSize;
                if(i-k+1<0) continue;
                if(kind<=maxLetters){
                    int cnt=++mp[s.substr(i-k+1,k)];
                    ans=max(ans,cnt);
                }
                int out=s[i-k+1]-'a';
                ccnt[out]--;
                if(ccnt[out]==0) kind--;
            }
            return ans;
        }
    };
