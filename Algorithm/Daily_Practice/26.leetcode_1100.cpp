class Solution {
    public:
        int numKLenSubstrNoRepeats(string s, int k) {
            int n = (int)s.size();
            if (k > n) return 0;
            vector<int> cnt(26, 0);
            int dup = 0, ans = 0;
            for (int i = 0; i < n; ++i) {
                int x = s[i] - 'a';
                if (++cnt[x] == 2) dup++;
                if (i < k - 1) continue;
                if (dup == 0) ans++;
                int y = s[i - k + 1] - 'a';
                if (cnt[y]-- == 2) dup--;
            }
            return ans;
        }
    };
