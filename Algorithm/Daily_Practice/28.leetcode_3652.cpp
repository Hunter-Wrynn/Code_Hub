// https://leetcode.cn/problems/stock-price-fluctuation-ii/description/
class Solution {
public:
    long long maxProfit(vector<int>& prices, vector<int>& strategy, int k) {
        long long total = 0, sum = 0;
        for (int i = 0; i < k / 2; ++i) {
            total += 1LL * prices[i] * strategy[i];
            sum -= 1LL * prices[i] * strategy[i];
        }
        for (int i = k / 2; i < k; ++i) {
            total += 1LL * prices[i] * strategy[i];
            sum += 1LL * prices[i] * (1 - strategy[i]);
        }
        long long ans = max(sum, 0LL);
        for (int i = k; i < (int)prices.size(); ++i) {
            total += 1LL * prices[i] * strategy[i];
            sum += 1LL * prices[i] * (1 - strategy[i]) + 1LL * prices[i - k] * strategy[i - k] - prices[i - k / 2];
            ans = max(sum, ans);
        }
        return total + ans;
    }
};
