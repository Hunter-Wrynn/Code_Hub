class Solution {
    public:
        vector<int> getAverages(vector<int>& nums, int k) {
            int n = nums.size();
            vector<int> ans(n, -1);
            if (k == 0) return nums;
            long long window = 2LL * k + 1;
            if (window > n) return ans;
    
            long long sum = 0;
            for (int i = 0; i < window; ++i) sum += nums[i];
    
            for (int i = k; i <= n - k - 1; ++i) {
                ans[i] = (int)(sum / window);
                if (i + k + 1 < n) {
                    sum -= nums[i - k];
                    sum += nums[i + k + 1];
                }
            }
            return ans;
        }
    };