class Solution {
    public:
        int minSwaps(vector<int>& nums) {
            int k=0;
            int n=nums.size();
            for(int i=0;i<n;++i){
                if(nums[i]==1) k++;
                nums.push_back(nums[i]);
            }
            if (k == 0) return 0;
            int as=1e5+1;
            int ak=0;
            for(int i=0;i<nums.size();++i){
                if(nums[i]==0) ak++;
                if(i-k+1<0) continue;
                as=min(as,ak);
                if(nums[i-k+1]==0)ak--;
            }
            return as;
        }
    };
