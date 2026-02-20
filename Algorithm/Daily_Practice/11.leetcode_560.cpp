class Solution {
    public:
        int subarraySum(vector<int>& nums, int k) {
            unordered_map<int,int> mp;
            mp[0]=1;
            int count=0;
            int pre=0;
            for(auto& x:nums){
                pre+=x;
                if(mp.find(pre-k)!=mp.end()) count+=mp[pre-k];
                mp[pre]++;
            }
            return count;
        }
    };