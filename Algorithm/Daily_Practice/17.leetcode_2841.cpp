class Solution {
    public:
        long long maxSum(vector<int>& nums, int m, int k) {
            unordered_map<int,int> mp;
            long long ans=0;
            long long as=0;
            for(int i=0;i<nums.size();++i){
                as+=nums[i];
                mp[nums[i]]++;
                if(i-k+1<0) continue;
                if(mp.size()>=m) ans=max(ans,as);
                as-=nums[i-k+1];
                
                if(--mp[nums[i-k+1]]==0) mp.erase(nums[i-k+1]);
            }   
            return ans;
        }
    };