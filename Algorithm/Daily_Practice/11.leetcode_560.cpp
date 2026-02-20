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

class Solution2 {
    public:
        int subarraySum(vector<int>& nums, int k) {
            int n = (int)nums.size();
            int offset = 1000 * n;
            int size = 2 * offset + 1;               // covers [-offset, +offset]
    
            vector<int> mp(size, 0);
            mp[offset] = 1;                          // pre = 0
    
            int count = 0;
            int pre = 0;
    
            for (int x : nums) {
                pre += x;
    
                int need = pre - k;
                int needIdx = need + offset;
                if (0 <= needIdx && needIdx < size) {
                    count += mp[needIdx];
                }
    
                int preIdx = pre + offset;
                // preIdx should always be in range given bounds, but keep safe if you want:
                if (0 <= preIdx && preIdx < size) {
                    mp[preIdx]++;
                }
            }
            return count;
        }
    };