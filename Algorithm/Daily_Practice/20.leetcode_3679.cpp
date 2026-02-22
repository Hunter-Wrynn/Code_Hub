class Solution {
    public:
        int minArrivalsToDiscard(vector<int>& arrivals, int w, int m) {
            unordered_map<int, int> cnt;
            int ans = 0;
            for (int i = 0; i < arrivals.size(); i++) {
                if (cnt[arrivals[i]] == m) { 
                    arrivals[i] = 0; // 丢弃 arrivals[i]
                    ans++;
                } else {
                    cnt[arrivals[i]]++;
                }
                // 左端点元素离开窗口，为下一个循环做准备
                int left = i + 1 - w;
                if (left >= 0) {
                    cnt[arrivals[left]]--;
                }
            }
            return ans;
        }
    };
    
    